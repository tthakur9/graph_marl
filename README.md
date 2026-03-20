Graph-Augmented MADDPG for Scalable MARL

Environment Setup:
`python3 -m venv .venv`
`source .venv/bin/activate`
`pip install --upgrade pip`
`pip install -r requirements.txt`

Train Baseline:
`python -m src.train`

Run on PACE-ICE:
Checkout `pace-ice` then run `sbatch job.sh`