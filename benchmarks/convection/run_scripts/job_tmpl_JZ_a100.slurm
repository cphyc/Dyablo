#!/bin/bash

# This is meant to be used on Jean Zay

#SBATCH -J dyablo_a100
#SBATCH -C a100 
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads> --gres=gpu:<nb_gpus>
#SBATCH -A <account>@a100
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH -o log.out


module purge
module load cpuarch/amd
module load cmake gcc/12.2.0 cuda/12.1.0 openmpi/4.1.5-cuda hdf5/1.12.0-mpi-cuda

export OMP_NUM_THREADS=<nb_threads>
export OMP_PROC_BIND=true

srun ./dyablo_a100 restart.ini
