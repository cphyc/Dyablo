#!/bin/bash

# This is meant to be used on AdAstra

#SBATCH -J dyablo_mi250
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads>
#SBATCH --account=<account>
#SBATCH --constraint=MI250
#SBATCH -t 1:00:00
#SBATCH --exclusive
#SBATCH -o log.out

modulr purge
module load PrgEnv-amd 
module load amd/5.5.1 cray-mpich cray-hdf5-parallel

export OMP_NUM_THREADS=<nb_threads>
export OMP_PROC_BIND=true 

srun ./dyablo_genoa restart.ini
