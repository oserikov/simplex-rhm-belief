"""Non-collapse sweep runner (pass 5): the skew x ambiguity grid.

Enumerates ``skew in {none,mid,high}`` x ``ambiguity in {none,mid,high}`` x
``draw in {0,1,2}`` -> 27 runs, invoking ``sweep.py --no-wandb --out-root
results/noncollapse`` per cell at the pinned published arch (n_layer=2,
n_embd=128, n_head=4, 4000 steps). The ``(none,none)`` cells reproduce the
pass-1 baseline anchor.

``draw`` is the rule-table index (passed as ``--grammar``); the master ``--seed``
is held at 0 so within a cell the 3 draws differ only in the grammar rules/probs.
Mirrors ``run_depth4.py``: deterministic dirs are the source of truth, resumable
(a cell whose ``config.json`` exists is skipped).

Run: ``uv run python run_noncollapse.py`` (``--dry-run`` to list the plan).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

N_EMBD, N_HEAD, STEPS, N_LAYER, SEED = 128, 4, 4000, 2, 0
SKEWS = ["none", "mid", "high"]
AMBIGUITY = {"none": 0.0, "mid": 0.3, "high": 0.6}
DRAWS = [0, 1, 2]
OUT_ROOT = Path("results/noncollapse")


def run_dir(skew: str, amb: float, draw: int) -> Path:
    base = f"g{draw:02d}_L{N_LAYER}_d{N_EMBD}_h{N_HEAD}_t{STEPS}_s{SEED}"
    if skew != "none" or amb > 0:
        base = f"sk{skew}_am{amb:g}_{base}"
    return OUT_ROOT / base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args()

    cells = [(sk, AMBIGUITY[am], draw)
             for sk in SKEWS for am in SKEWS for draw in DRAWS]
    assert len(cells) == 27, f"expected 27 runs, got {len(cells)}"
    print(f"noncollapse sweep: {len(SKEWS)} skew x {len(SKEWS)} ambiguity "
          f"x {len(DRAWS)} draws = {len(cells)} runs (L=3, arch pinned)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "run_noncollapse.log"
    done = sum((run_dir(sk, am, d) / "config.json").exists() for sk, am, d in cells)
    print(f"{done}/{len(cells)} cells already complete; {len(cells) - done} to run")
    if args.dry_run:
        for sk, am, d in cells:
            print(f"  skew={sk:4s} amb={am:g} draw={d} -> {run_dir(sk, am, d).name}")
        return

    t_start = time.time()
    skew_of_amb = {v: k for k, v in AMBIGUITY.items()}
    with log.open("a") as lf:
        lf.write(f"\n=== run_noncollapse start {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
        for i, (sk, am, draw) in enumerate(cells, 1):
            d = run_dir(sk, am, draw)
            tag = f"[{i}/{len(cells)}] skew={sk} amb={am:g}({skew_of_amb[am]}) draw={draw}"
            if (d / "config.json").exists():
                print(f"SKIP {tag} (exists)")
                continue
            cmd = [
                sys.executable, "sweep.py", "--no-wandb",
                "--out-root", str(OUT_ROOT),
                "--skew", sk, "--ambiguity", str(am),
                "--grammar", str(draw), "--seed", str(SEED),
                "--n-layer", str(N_LAYER), "--n-embd", str(N_EMBD),
                "--n-head", str(N_HEAD), "--steps", str(STEPS),
            ]
            t0 = time.time()
            print(f"RUN  {tag}", flush=True)
            lf.write(f"{tag} start {time.strftime('%H:%M:%S')}\n"); lf.flush()
            r = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            tail = "\n".join(
                ln for ln in r.stdout.splitlines()
                if ln.startswith(("FINAL", "DONE", "non-collapse"))
            )
            if r.returncode != 0:
                print(f"FAIL {tag} ({dt:.0f}s)\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
                lf.write(f"{tag} FAILED rc={r.returncode}\n{r.stderr[-1500:]}\n"); lf.flush()
                continue
            print(f"  {tail}  ({dt:.0f}s)")
            lf.write(f"{tail}  ({dt:.0f}s)\n"); lf.flush()

    elapsed = time.time() - t_start
    n_done = sum((run_dir(sk, am, d) / "config.json").exists() for sk, am, d in cells)
    print(f"\nrun_noncollapse done: {n_done}/{len(cells)} complete ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
