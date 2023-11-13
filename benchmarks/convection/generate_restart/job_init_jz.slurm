#!/bin/bash
#SBATCH --job-name=dyablo_bench_init
#SBATCH -C v100-32g 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10  
#SBATCH --hint=nomultithread  # Pas d'hyperthreading par défaut. Discutable ?
#SBATCH --time=06:00:00
#SBATCH --output=dyablo.out # nom du fichier de sortie
#SBATCH --error=dyablo.out  # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A jza@v100

module purge
module load cmake/ gcc/8.5.0 cuda/12.1.0 openmpi/4.0.5-cuda hdf5/1.12.0-mpi-cuda

export OMP_PROC_BIND=true

cd 1_run_prep
rm *.h5
srun ../dyablo_v100 run_prep.ini

cd ../2_refine_1
rm *.h5
cp ../1_run_prep/restart_C91*.h5 restart.h5
srun ../dyablo_v100 refine_1.ini

cd ../3_refine_2
rm *.h5
cp ../1_run_prep/restart_C91*.h5 restart.h5
srun ../dyablo_v100 refine_2.ini
