"""
Scaling experiment: baseline vs GCN vs GAT at increasing agent counts.

Tests the hypothesis that graph communication helps at scale.
Two conditions:
  - equal:   n_chasers == n_evaders (symmetric teams)
  - unequal: 3:1 ratio (mirrors original default)

Graph params fixed at conf/graph.yaml defaults (n_layers=2, n_heads=4, radius=null).

Job counts (3 models x 3 seeds):
  equal   : 4 scale points x 9 =  36 jobs
  unequal : 4 scale points x 9 =  36 jobs
  total                         =  72 jobs

Usage:
  python scaling.py              # submit all jobs
  python scaling.py --dry-run
  python scaling.py --model baseline|gcn|gat
  python scaling.py --rerun-failed
"""

from __future__ import annotations
import argparse
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


def is_complete(model: str, label: str, seed: int) -> bool:
    pattern = Path("runs") / "scaling" / model / label / f"*_seed{seed}" / "metrics.csv"
    return bool(list(Path(".").glob(str(pattern))))


def generate_config(out_dir: Path, model: str, condition: str,
                    agents: dict, fixed: dict) -> tuple[Path, str]:
    cfg = OmegaConf.load(BASE_CONFIGS[model])
    apply_fixed(cfg, fixed)

    if model in ("gcn", "gat"):
        OmegaConf.update(cfg, "graph.backbone", model, merge=True)

    n_ch = agents["env.n_chasers"]
    n_ev = agents["env.n_evaders"]
    for key, val in agents.items():
        OmegaConf.update(cfg, key, val, merge=True)

    label = f"ch{n_ch}_ev{n_ev}"
    OmegaConf.update(cfg, "run_group", f"scaling/{model}/{condition}/{label}", merge=True)

    model_dir = out_dir / model / condition
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = model_dir / f"{label}.yaml"
    OmegaConf.save(cfg, cfg_path)
    return cfg_path, label


def submit(model: str, cfg_path: Path, seed: int, dry_run: bool) -> str:
    cmd = ["sbatch", "job_scaling.sh", model, str(cfg_path), str(seed)]
    if dry_run:
        return "dry-run"
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", choices=["baseline", "gcn", "gat"], default=None)
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Skip configs that already have a completed metrics.csv")
    args = parser.parse_args()

    cfg      = OmegaConf.load("conf/scaling.yaml")
    fixed    = OmegaConf.to_container(cfg.fixed, resolve=True)
    seeds    = list(cfg.seeds)
    models   = list(cfg.models) if args.model is None else [args.model]
    out_dir  = Path("conf/scaling")

    total = 0
    for condition in ("equal", "unequal"):
        print(f"\n=== {condition} ===")
        scale_points = OmegaConf.to_container(cfg[condition], resolve=True)
        for model in models:
            for agents in scale_points:
                cfg_path, label = generate_config(out_dir, model, condition, agents, fixed)
                for seed in seeds:
                    if args.rerun_failed and is_complete(model, f"{condition}/{label}", seed):
                        print(f"  [{model}/{condition}/{label}] seed={seed}: skip")
                        continue
                    msg = submit(model, cfg_path, seed, args.dry_run)
                    print(f"  [{model}/{condition}/{label}] seed={seed}: {msg}")
                    total += 1

    print(f"\nTotal jobs {'(dry-run) ' if args.dry_run else ''}submitted: {total}")


if __name__ == "__main__":
    main()
