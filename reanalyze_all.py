"""Re-run analyze_run over existing sweep dirs (reuses saved probe_data.npz, no
retrain). Used when only the metric/figure computation changed."""

import glob
import os

from analyze import analyze_run

dirs = sorted(glob.glob("results/sweep/g*"))
print(f"re-analyzing {len(dirs)} run dirs")
for i, d in enumerate(dirs):
    if not os.path.exists(os.path.join(d, "probe_data.npz")):
        print(f"SKIP {d} (no probe_data.npz)")
        continue
    a = analyze_run(art_dir=d, res_dir=d)
    r = a["blooming_radius_by_position"]
    print(f"[{i + 1}/{len(dirs)}] {os.path.basename(d)} "
          f"radius k1={r[0]:.3f} k8={r[-1]:.3f} root_r2={a['sanity']['root_r2']:.3f}")
