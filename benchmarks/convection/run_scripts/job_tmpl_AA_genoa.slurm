#!/bin/bash

# This is meant to be used on AdAstra

#SBATCH -J dyablo_cpu
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads>
#SBATCH --account=<account>
#SBATCH --constraint=GENOA
#SBATCH -t 1:00:00
#SBATCH --exclusive
#SBATCH -o log.out

module purge
module load cpe/23.05
module load craype-x86-genoa
module load PrgEnv-cray
module load cray-hdf5-parallel

export OMP_NUM_THREADS=<nb_threads>
export OMP_PROC_BIND=true 

export MPICH_GPU_SUPPORT_ENABLED=1

srun --ntasks=<nb_mpi_tot> --ntasks-per-node=<nb_mpi> --cpus-per-task=<nb_threads> --gpu-bind=closest ./dyablo_genoa restart.ini
