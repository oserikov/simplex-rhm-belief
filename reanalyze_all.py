"""Re-run analyze_run over existing run dirs (reuses saved probe_data.npz, no
retrain). Used when only the metric/figure computation changed (e.g. adding RMSE).

Covers every pass that reported R2: the canonical run (artifacts/, results/refrun/)
and the four sweeps (results/{sweep,arch,depth4,noncollapse}/*)."""

import glob
import os

from analyze import analyze_run

dirs = ["artifacts", "results/refrun"]
for sub in ("sweep", "arch", "depth4", "noncollapse"):
    dirs += sorted(glob.glob(f"results/{sub}/*"))
dirs = [d for d in dirs if os.path.isdir(d)]
print(f"re-analyzing {len(dirs)} run dirs")

n_ok = 0
for i, d in enumerate(dirs):
    if not os.path.exists(os.path.join(d, "probe_data.npz")):
        print(f"SKIP {d} (no probe_data.npz)")
        continue
    a = analyze_run(art_dir=d, res_dir=d)
    s = a["sanity"]
    n_ok += 1
    print(f"[{i + 1}/{len(dirs)}] {os.path.basename(d)} "
          f"root_r2={s['root_r2']:.3f} root_rmse={s['root_rmse']:.3f} "
          f"deepest_rmse={s['deepest_rmse']:.3f}")
print(f"done: re-analyzed {n_ok} dirs")
