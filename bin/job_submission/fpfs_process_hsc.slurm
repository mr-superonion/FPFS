#!/bin/bash
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 28
#SBATCH --cpus-per-task=1
#SBATCH --time 24:00:00
#SBATCH --job-name xjob
#SBATCH --output fpfs_job-out
#SBATCH --error fpfs_job-err
#SBATCH --mem-per-cpu=4G

export SLURM_STEP_NODELIST=$SLURM_NODELIST
echo $SLURM_PROCID
source /hildafs/projects/phy200017p/share/ana/setupLsstim1.sh
source fpfs_config

cd $PWD
mpirun -n 56 fpfs_process_sim.py --config ./config_procsim.ini --minId 0 --maxId 4000 --mpi
