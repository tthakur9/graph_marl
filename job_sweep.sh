#!/bin/bash
#SBATCH -J sweep
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH -t 2:00:00
#SBATCH -o logs/sweep_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bjayaraman9@gatech.edu

CONFIG=$1
SEED=$2

cd $SLURM_SUBMIT_DIR
mkdir -p logs
module load cuda
source .venv/bin/activate
python -m src.train --config "$CONFIG" --seed "$SEED"
