"""Grammar-sweep entry point: one (grammar, seed, arch) run, fully tracked.

Resolves its config from **``wandb.config``** when launched by a W&B agent
(``wandb agent <id>``) and falls back to **``argparse``** when run standalone
(``uv run python sweep.py --grammar 3 --seed 1``). Either way it:

  build grammar + model -> train -> analyze -> log metrics to W&B -> write a
  deterministic, human-readable artifact dir + ``config.json`` manifest.

The artifact dir is independent of any W&B id, so every run is recoverable for
the paper even with no network:

    results/sweep/g{grammar:02d}_L{n_layer}_d{n_embd}_h{n_head}_s{seed}/
      config.json grammar.npz model.pt probe_data.npz analysis.json

``--no-wandb`` (or ``WANDB_MODE=offline``) runs the identical pipeline with no
network, for the seconds-level quick-failure checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from analyze import analyze_run
from train import train

# Open config: set via env or edit these defaults before the real online run.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "simplex-rhm-grammar-sweep")

DEFAULT_SWEEP_ROOT = Path("results/sweep")

# Fixed RHM constraints for this pass (never swept).
RHM_FIXED = dict(s=2, L=3, v=8, m=2)


def run_dir_name(cfg: SimpleNamespace) -> str:
    return (f"g{cfg.grammar:02d}_L{cfg.n_layer}_d{cfg.n_embd}"
            f"_h{cfg.n_head}_t{cfg.steps}_s{cfg.seed}")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_train_args(cfg: SimpleNamespace) -> SimpleNamespace:
    """Map the resolved sweep config onto train.train()'s arg namespace.

    ``grammar`` (the swept rule-table index) -> ``grammar_seed``; the master
    ``seed`` drives model init + data + probe sampling + split.
    """
    return SimpleNamespace(
        s=RHM_FIXED["s"], L=getattr(cfg, "rhm_L", RHM_FIXED["L"]),
        v=RHM_FIXED["v"], m=RHM_FIXED["m"],
        grammar_seed=cfg.grammar, seed=cfg.seed,
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        lr=cfg.lr, steps=cfg.steps, batch_size=cfg.batch_size,
        test_frac=cfg.test_frac, n_probe=cfg.n_probe, log_every=cfg.log_every,
    )


def resolve_config() -> tuple[SimpleNamespace, bool, object, Path]:
    """Resolve config from wandb.config (under an agent) or argparse.

    Returns (cfg, use_wandb, wandb_run_or_None, out_root).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--grammar", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-probe", type=int, default=400)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--out-root", type=str, default=str(DEFAULT_SWEEP_ROOT),
                   help="root dir for artifact dirs (this pass: results/arch)")
    p.add_argument("--rhm-L", type=int, default=RHM_FIXED["L"],
                   help="RHM tree depth (default 3; pass 4 uses 4 -> context d=s^L=16)")
    p.add_argument("--no-wandb", action="store_true",
                   help="run the identical pipeline with no network")
    args = p.parse_args()

    # Fail fast on invalid head splits before any model/data work.
    if args.n_embd % args.n_head != 0:
        p.error(f"n_embd % n_head != 0: n_embd={args.n_embd} not divisible by "
                f"n_head={args.n_head} (GPT2 requires n_embd divisible by n_head)")

    use_wandb = not args.no_wandb and os.environ.get("WANDB_MODE") != "disabled"

    run = None
    cfg_dict = dict(
        grammar=args.grammar, seed=args.seed,
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        lr=args.lr, steps=args.steps, batch_size=args.batch_size,
        test_frac=args.test_frac, n_probe=args.n_probe, log_every=args.log_every,
        rhm_L=args.rhm_L,
    )

    if use_wandb:
        import wandb
        # Under `wandb agent`, wandb.init() pulls swept values into wandb.config
        # (overriding our argparse defaults); standalone it just records them.
        run = wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT,
            config=cfg_dict, reinit=True,
        )
        cfg_dict = dict(wandb.config)

    cfg = SimpleNamespace(**cfg_dict)
    return cfg, use_wandb, run, Path(args.out_root)


def main() -> None:
    cfg, use_wandb, run, out_root = resolve_config()
    out_dir = out_root / run_dir_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== sweep run -> {out_dir} (wandb={'on' if use_wandb else 'off'}) ===")

    t0 = time.time()
    train_args = build_train_args(cfg)
    summary = train(train_args, out_dir=out_dir)
    analysis = analyze_run(art_dir=out_dir, res_dir=out_dir)
    elapsed = time.time() - t0

    # ---- flat metrics for W&B + the sweep objective --------------------
    level_r2 = analysis["latent_level_r2"]
    sanity = analysis["sanity"]
    metrics = {
        "root_r2": level_r2["root_L0"],
        "deepest_r2": sanity["deepest_r2"],
        "loss_gap_closed": sanity["loss_gap_closed"],
        "test_ce": sanity["test_ce"],
        "layer0_r2": analysis["layer_r2"][0],
        "layer_final_r2": analysis["layer_r2"][-1],
        "elapsed_sec": elapsed,
    }
    for name, r2 in level_r2.items():
        metrics[f"r2_{name}"] = r2

    # ---- deterministic manifest ----------------------------------------
    manifest = {
        "config": vars(cfg),
        "rhm_fixed": {**RHM_FIXED, "L": getattr(cfg, "rhm_L", RHM_FIXED["L"])},
        "metrics": metrics,
        "wandb_run_id": (run.id if run is not None else None),
        "wandb_run_path": (f"{run.entity}/{run.project}/{run.id}"
                           if run is not None else None),
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "train_summary": summary,
    }
    (out_dir / "config.json").write_text(json.dumps(manifest, indent=2))

    if use_wandb:
        import wandb
        wandb.log(metrics)
        wandb.summary.update(metrics)
        wandb.finish()

    print(f"DONE {out_dir.name}: root_r2={metrics['root_r2']:.4f} "
          f"deepest_r2={metrics['deepest_r2']:.4f} "
          f"loss_gap_closed={metrics['loss_gap_closed']:.4f} "
          f"({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
