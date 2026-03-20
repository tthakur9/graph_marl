#!/bin/bash
#SBATCH -J graph_marl
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH -t4:00:00
#SBATCH --array=0-4
#SBATCH -o logs/seed%a.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjayaraman9@gatech.edu

cd $SLURM_SUBMIT_DIR
mkdir -p logs
source .venv/bin/activate
srun python -m src.train --seed $SLURM_ARRAY_TASK_ID
