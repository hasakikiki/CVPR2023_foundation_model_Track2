#!/bin/bash
###==========PARAMETER!!!==========
#SBATCH -J slurm_test
#SBATCH -p a100
#SBATCH --cpus-per-task=10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o OneForAll/logs/slurm_test_result.log
###=======PARAMETER END!!!=========

# env
module load cuda/11.6
eval "$(conda shell.bash hook)"
conda activate paddle

# command
python OneForAll/paddle_matrix_multiplication.py
