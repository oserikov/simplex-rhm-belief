#!/bin/bash
# Run the full grammar x seed sweep at pinned arch, full budget. Deterministic
# artifact dirs are the source of truth; --no-wandb keeps it network-free.
set -e
cd "$(dirname "$0")"
LOG=results/sweep/run_all.log
: > "$LOG"
echo "sweep start $(date)" | tee -a "$LOG"
for g in 0 1 2 3 4 5 6 7 8 9; do
  for s in 0 1 2; do
    echo "=== grammar=$g seed=$s $(date +%H:%M:%S) ===" | tee -a "$LOG"
    uv run python sweep.py --no-wandb --grammar "$g" --seed "$s" --steps 4000 \
      2>&1 | grep -E "FINAL|DONE" | tee -a "$LOG"
  done
done
echo "sweep done $(date)" | tee -a "$LOG"
echo "completed runs: $(ls -d results/sweep/g*/ | wc -l)" | tee -a "$LOG"
