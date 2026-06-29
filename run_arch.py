"""Architecture-sweep runner (pass 3): the OAT union of arch axes.

Enumerates the 11 one-axis-at-a-time (OAT) configs around the published pass-1
baseline (``n_layer=2, n_embd=128, n_head=4, steps=4000``), crossed with
``grammar in {0,1,2}`` and master ``seed in {0,1,2}`` -> 99 runs, and invokes
``sweep.py --no-wandb --out-root results/arch`` per cell (deterministic dirs are
the source of truth; OAT is a *union of axis-slices*, not a cartesian grid, so
it is driven by this explicit runner rather than a W&B grid sweep).

Resumable: a cell whose ``config.json`` already exists is skipped. Run with
``uv run python run_arch.py`` (optionally ``--dry-run`` to list the plan).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Published pass-1 baseline (the shared center point of all four axis slices).
BASE = dict(n_layer=2, n_embd=128, n_head=4, steps=4000)

# One-axis-at-a-time values (each axis varies, the other three held at BASE).
AXES = {
    "n_embd": [16, 64, 128, 256],
    "n_layer": [1, 2, 3, 4],
    "n_head": [1, 2, 4, 8],
    "steps": [4000, 16000],
}

GRAMMARS = [0, 1, 2]
SEEDS = [0, 1, 2]
OUT_ROOT = Path("results/arch")


def oat_configs() -> list[dict]:
    """The 11 distinct OAT arch configs (baseline dedup across the four slices)."""
    seen: set[tuple] = set()
    configs: list[dict] = []
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(BASE)
            cfg[axis] = v
            key = (cfg["n_layer"], cfg["n_embd"], cfg["n_head"], cfg["steps"])
            if key in seen:
                continue
            seen.add(key)
            configs.append(cfg)
    return configs


def run_dir(cfg: dict, grammar: int, seed: int) -> Path:
    return OUT_ROOT / (
        f"g{grammar:02d}_L{cfg['n_layer']}_d{cfg['n_embd']}"
        f"_h{cfg['n_head']}_t{cfg['steps']}_s{seed}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args()

    configs = oat_configs()
    cells = [(c, g, s) for c in configs for g in GRAMMARS for s in SEEDS]
    assert len(configs) == 11, f"expected 11 OAT configs, got {len(configs)}"
    assert len(cells) == 99, f"expected 99 runs, got {len(cells)}"

    print(f"OAT sweep: {len(configs)} configs x {len(GRAMMARS)} grammars "
          f"x {len(SEEDS)} seeds = {len(cells)} runs")
    for c in configs:
        print(f"  config L{c['n_layer']} d{c['n_embd']} h{c['n_head']} t{c['steps']}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "run_arch.log"
    done = sum((run_dir(c, g, s) / "config.json").exists() for c, g, s in cells)
    print(f"{done}/{len(cells)} cells already complete; "
          f"{len(cells) - done} to run")

    if args.dry_run:
        return

    t_start = time.time()
    with log.open("a") as lf:
        lf.write(f"\n=== run_arch start {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        for i, (cfg, g, s) in enumerate(cells, 1):
            d = run_dir(cfg, g, s)
            tag = (f"[{i}/{len(cells)}] g{g} s{s} "
                   f"L{cfg['n_layer']} d{cfg['n_embd']} h{cfg['n_head']} t{cfg['steps']}")
            if (d / "config.json").exists():
                print(f"SKIP {tag} (exists)")
                continue
            cmd = [
                sys.executable, "sweep.py", "--no-wandb",
                "--out-root", str(OUT_ROOT),
                "--grammar", str(g), "--seed", str(s),
                "--n-layer", str(cfg["n_layer"]), "--n-embd", str(cfg["n_embd"]),
                "--n-head", str(cfg["n_head"]), "--steps", str(cfg["steps"]),
            ]
            t0 = time.time()
            print(f"RUN  {tag}", flush=True)
            lf.write(f"{tag} start {time.strftime('%H:%M:%S')}\n"); lf.flush()
            r = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            tail = "\n".join(
                ln for ln in r.stdout.splitlines() if ln.startswith(("FINAL", "DONE"))
            )
            if r.returncode != 0:
                print(f"FAIL {tag} ({dt:.0f}s)\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                lf.write(f"{tag} FAILED rc={r.returncode}\n{r.stderr[-2000:]}\n")
                lf.flush()
                continue
            print(f"  {tail}  ({dt:.0f}s)")
            lf.write(f"{tail}  ({dt:.0f}s)\n"); lf.flush()

    elapsed = time.time() - t_start
    n_done = sum((run_dir(c, g, s) / "config.json").exists() for c, g, s in cells)
    print(f"\nrun_arch done: {n_done}/{len(cells)} complete "
          f"({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
