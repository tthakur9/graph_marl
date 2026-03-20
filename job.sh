#!/bin/bash
#SBATCH -J graph_marl
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH -t4:00:00
#SBATCH -o output.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bjayaraman9@gatech.edu

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
srun python -m src.train
