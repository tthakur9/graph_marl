#!/bin/bash
#SBATCH -J scaling
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH -t 6:00:00
#SBATCH -o logs/scaling_%j.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bjayaraman9@gatech.edu

MODEL=$1
CONFIG=$2
SEED=$3

cd $SLURM_SUBMIT_DIR
mkdir -p logs
module load cuda
source .venv/bin/activate

if [ "$MODEL" = "baseline" ]; then
    python -m src.train --config "$CONFIG" --seed "$SEED"
else
    python -m src.train_graph --config "$CONFIG" --seed "$SEED"
fi
