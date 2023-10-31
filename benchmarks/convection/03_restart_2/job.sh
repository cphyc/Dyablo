#!/bin/bash
#SBATCH --job-name=Dyablo-base_run     # nom du job
#SBATCH -C a100                 # decommenter pour reserver uniquement des GPU V100 32 Go
#SBATCH --nodes=1                    # nombre de noeud
#SBATCH --ntasks-per-node=8          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:8                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=10:00:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=dyablo.out # nom du fichier de sortie
#SBATCH --error=dyablo.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A jza@a100
##SBATCH --qos=qos_gpu-dev

module purge
module load cpuarch/amd
module load cmake gcc/8.5.0 cuda/12.1.0 openmpi/4.0.5-cuda hdf5/1.12.0-mpi-cuda

# Echo des commandes lancees
set -x
 
srun ./dyablo_a100 restart.ini
