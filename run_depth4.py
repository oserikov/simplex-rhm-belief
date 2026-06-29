"""Depth-4 grammar-sweep runner (pass 4): the L=4 replication family.

Enumerates ``grammar in 0..9`` x master ``seed in 0..2`` x ``n_layer in {2,3}``
-> 60 runs, and invokes ``sweep.py --no-wandb --rhm-L 4 --out-root
results/depth4`` per cell. Mirrors pass-2 ``run_all.sh`` / pass-3
``run_arch.py``; the deterministic dirs are the source of truth.

L=4 runs write to a separate root ``results/depth4`` so their dir names (which
encode arch but NOT tree depth) never collide with the L=3 ``results/sweep``
(pass 2) or ``results/arch`` (pass 3).

Resumable: a cell whose ``config.json`` already exists is skipped. Run with
``uv run python run_depth4.py`` (optionally ``--dry-run`` to list the plan).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Fixed arch for this pass (only n_layer varies, as a 2-cell depth mini-axis).
N_EMBD, N_HEAD, STEPS = 128, 4, 4000
RHM_L = 4

GRAMMARS = list(range(10))
SEEDS = [0, 1, 2]
N_LAYERS = [2, 3]
OUT_ROOT = Path("results/depth4")


def run_dir(grammar: int, seed: int, n_layer: int) -> Path:
    return OUT_ROOT / (
        f"g{grammar:02d}_L{n_layer}_d{N_EMBD}_h{N_HEAD}_t{STEPS}_s{seed}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args()

    cells = [(g, s, nl) for g in GRAMMARS for s in SEEDS for nl in N_LAYERS]
    assert len(cells) == 60, f"expected 60 runs, got {len(cells)}"

    print(f"depth4 sweep: {len(GRAMMARS)} grammars x {len(SEEDS)} seeds "
          f"x {len(N_LAYERS)} n_layer = {len(cells)} runs (rhm L={RHM_L})")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "run_depth4.log"
    done = sum((run_dir(g, s, nl) / "config.json").exists() for g, s, nl in cells)
    print(f"{done}/{len(cells)} cells already complete; {len(cells) - done} to run")

    if args.dry_run:
        return

    t_start = time.time()
    with log.open("a") as lf:
        lf.write(f"\n=== run_depth4 start {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        for i, (g, s, nl) in enumerate(cells, 1):
            d = run_dir(g, s, nl)
            tag = (f"[{i}/{len(cells)}] g{g} s{s} L{nl} "
                   f"d{N_EMBD} h{N_HEAD} t{STEPS}")
            if (d / "config.json").exists():
                print(f"SKIP {tag} (exists)")
                continue
            cmd = [
                sys.executable, "sweep.py", "--no-wandb", "--rhm-L", str(RHM_L),
                "--out-root", str(OUT_ROOT),
                "--grammar", str(g), "--seed", str(s),
                "--n-layer", str(nl), "--n-embd", str(N_EMBD),
                "--n-head", str(N_HEAD), "--steps", str(STEPS),
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
    n_done = sum((run_dir(g, s, nl) / "config.json").exists() for g, s, nl in cells)
    print(f"\nrun_depth4 done: {n_done}/{len(cells)} complete ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
