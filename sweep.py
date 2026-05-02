"""
Reads conf/sweep.yaml, generates one config per hyperparameter combo,
and submits one SLURM job per (combo, seed).
"""

import itertools
import subprocess
from pathlib import Path
from omegaconf import OmegaConf


def main():
    sweep_cfg = OmegaConf.load("conf/sweep.yaml")
    base_cfg = OmegaConf.load(sweep_cfg.base)

    grid_keys = list(sweep_cfg.grid.keys())
    grid_values = [list(sweep_cfg.grid[k]) for k in grid_keys]

    combo_dir = Path("conf/sweep")
    combo_dir.mkdir(parents=True, exist_ok=True)

    for combo in itertools.product(*grid_values):
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

        label_parts = []
        for key, val in zip(grid_keys, combo):
            OmegaConf.update(cfg, key, val)
            short_key = key.split(".")[-1]
            label_parts.append(f"{short_key}{val}")
        label = "_".join(label_parts)

        OmegaConf.update(cfg, "run_group", f"sweep/{label}")

        combo_path = combo_dir / f"{label}.yaml"
        OmegaConf.save(cfg, combo_path)
        print(f"Generated {combo_path}")

        for seed in sweep_cfg.seeds:
            result = subprocess.run(
                ["sbatch", "job_sweep.sh", str(combo_path), str(seed)],
                capture_output=True, text=True,
            )
            print(f"  seed={seed}: {result.stdout.strip() or result.stderr.strip()}")


if __name__ == "__main__":
    main()
