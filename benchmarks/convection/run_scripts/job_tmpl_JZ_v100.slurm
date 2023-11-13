#!/bin/bash

# This is meant to be used on Jean Zay

#SBATCH -J dyablo_v100
#SBATCH -C v100-32g 
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads> --gres=gpu:<nb_gpus>
#SBATCH -A <account>@v100
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH -o log.out


module purge
module load cmake/ gcc/8.5.0 cuda/12.1.0 openmpi/4.0.5-cuda hdf5/1.12.0-mpi-cuda

export OMP_NUM_THREADS=<nb_threads>
export OMP_PROC_BIND=true

srun ./dyablo_v100 restart.ini
