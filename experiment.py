"""
Full comparison experiment: baseline vs GCN vs GAT.

All models sweep over env vars (n_chasers, n_evaders, n_obstacles).
GCN and GAT additionally sweep over graph architecture params,
crossed with the env grid.

Fixed across all runs: lr=3e-4, n_opt_steps=200  (best from LR sweep)

Job counts (3 seeds each):
  baseline :  18 env configs                    =   54 jobs
  GCN      :  18 env x  9 graph configs         =  486 jobs
  GAT      :  18 env x 27 graph configs         = 1458 jobs
  total                                         = 1998 jobs

Usage:
  python experiment.py              # submit all jobs
  python experiment.py --dry-run    # print sbatch commands without submitting
  python experiment.py --model baseline|gcn|gat  # submit one model only
"""

from __future__ import annotations
import argparse
import itertools
import subprocess
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


BASE_CONFIGS = {
    "baseline": "conf/baseline.yaml",
    "gcn":      "conf/graph.yaml",
    "gat":      "conf/graph.yaml",
}


def apply_fixed(cfg: DictConfig, fixed: dict) -> None:
    for key, val in fixed.items():
        OmegaConf.update(cfg, key, val, merge=True)


def make_label(keys: list[str], combo: tuple) -> str:
    parts = []
    for key, val in zip(keys, combo):
        short = key.split(".")[-1]
        val_str = "none" if val is None else str(val)
        parts.append(f"{short}{val_str}")
    return "_".join(parts)


def generate_configs(
    out_dir: Path,
    model: str,
    grids: list[dict],
    fixed: dict,
) -> list[tuple[Path, str]]:
    """
    Cross-product of all grids, write one yaml per combo.
    grids: list of dicts to combine (e.g. [env_grid, graph_grid]).
    Returns list of (cfg_path, label).
    """
    base_cfg = OmegaConf.load(BASE_CONFIGS[model])
    apply_fixed(base_cfg, fixed)
    if model in ("gcn", "gat"):
        OmegaConf.update(base_cfg, "graph.backbone", model, merge=True)

    # Merge all grids into one flat key/value structure
    combined = {}
    for g in grids:
        combined.update(g)

    keys   = list(combined.keys())
    values = [list(combined[k]) for k in keys]

    model_dir = out_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for combo in itertools.product(*values):
        cfg   = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        label = make_label(keys, combo)

        for key, val in zip(keys, combo):
            OmegaConf.update(cfg, key, val, merge=True)

        OmegaConf.update(cfg, "run_group", f"experiment/{model}/{label}", merge=True)

        cfg_path = model_dir / f"{label}.yaml"
        OmegaConf.save(cfg, cfg_path)
        results.append((cfg_path, label))

    return results


def is_complete(model: str, label: str, seed: int) -> bool:
    """Return True if a metrics.csv already exists for this (model, label, seed)."""
    pattern = Path("runs") / "experiment" / model / label / f"*_seed{seed}" / "metrics.csv"
    return bool(list(Path(".").glob(str(pattern))))


def submit(model: str, cfg_path: Path, seed: int, dry_run: bool) -> str:
    cmd = ["sbatch", "job_experiment.sh", model, str(cfg_path), str(seed)]
    if dry_run:
        return "dry-run"
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def run_model(model: str, grids: list[dict], fixed: dict, seeds: list[int],
              out_dir: Path, dry_run: bool, rerun_failed: bool) -> int:
    configs = generate_configs(out_dir, model, grids, fixed)
    n_jobs  = 0
    for cfg_path, label in configs:
        for seed in seeds:
            if rerun_failed and is_complete(model, label, seed):
                print(f"  [{model}/{label}] seed={seed}: skip (already complete)")
                continue
            msg = submit(model, cfg_path, seed, dry_run)
            print(f"  [{model}/{label}] seed={seed}: {msg}")
            n_jobs += 1
    return n_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    parser.add_argument("--model", choices=["baseline", "gcn", "gat"],
                        default=None, help="Submit jobs for one model only")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Skip configs that already have a completed metrics.csv")
    args = parser.parse_args()

    exp_cfg  = OmegaConf.load("conf/experiment.yaml")
    fixed    = OmegaConf.to_container(exp_cfg.fixed,    resolve=True)
    env_grid = OmegaConf.to_container(exp_cfg.env_grid, resolve=True)
    gcn_grid = OmegaConf.to_container(exp_cfg.gcn_grid, resolve=True)
    gat_grid = OmegaConf.to_container(exp_cfg.gat_grid, resolve=True)
    seeds    = list(exp_cfg.seeds)
    out_dir  = Path("conf/experiment")

    total = 0
    run_all = args.model is None

    if run_all or args.model == "baseline":
        print("=== Baseline ===")
        total += run_model("baseline", [env_grid], fixed, seeds, out_dir, args.dry_run, args.rerun_failed)

    if run_all or args.model == "gcn":
        print("\n=== GCN ===")
        total += run_model("gcn", [env_grid, gcn_grid], fixed, seeds, out_dir, args.dry_run, args.rerun_failed)

    if run_all or args.model == "gat":
        print("\n=== GAT ===")
        total += run_model("gat", [env_grid, gat_grid], fixed, seeds, out_dir, args.dry_run, args.rerun_failed)

    print(f"\nTotal jobs {'(dry-run) ' if args.dry_run else ''}submitted: {total}")


if __name__ == "__main__":
    main()
