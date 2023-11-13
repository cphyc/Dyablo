#!/bin/bash

# This is meant to be used on Jean Zay

#SBATCH -J dyablo_cpu
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads>
#SBATCH -A <account>@cpu
#SBATCH -t 1:00:00
#SBATCH --exclusive
#SBATCH -o log.out

module purge
module load cmake/ gcc/12.2.0 openmpi/4.0.5 hdf5/1.12.0-mpi

export OMP_NUM_THREADS=<nb_threads>
export OMP_PROC_BIND=true 

srun ./dyablo_csl restart.ini
