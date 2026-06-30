# Figure: loss_curve.png
# Message: The transformer converges to the true RHM predictor -- held-out
#   next-token cross-entropy drops from the uniform baseline toward the
#   Bayes-optimal floor over training, closing ~89% of the gap.
# Data: results/refrun/train_summary.json -- a seeded reproduction of the pass-1
#   canonical run (grammar seed 0, pinned arch n_layer=2/n_embd=128/n_head=4,
#   4000 steps, test_frac 0.1) with the per-step loss history saved
#   (`loss_history`: [step, train_minibatch_CE, held-out_test_CE]).
# Type: training curve (test CE vs step) with uniform + Bayes-floor reference lines.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _style import OUTDIR, ROOT, apply_style

apply_style()

with open(os.path.join(ROOT, "results/refrun/train_summary.json")) as f:
    S = json.load(f)
hist = np.array(S["loss_history"], dtype=float)  # (n, 3): step, train, test
steps, train_ce, test_ce = hist[:, 0], hist[:, 1], hist[:, 2]
uniform = float(S["uniform_baseline"])
floor = float(S["bayes_optimal_mean"])
final = float(S["final_test_loss"])
gap_closed = (uniform - final) / (uniform - floor)
print(f"final test CE {final:.4f}  uniform {uniform:.4f}  floor {floor:.4f}  "
      f"gap closed {gap_closed:.3f}")
assert np.isfinite(hist).all() and len(hist) > 10, "loss history missing/short"

pal = sns.color_palette("colorblind")
fig, ax = plt.subplots(figsize=(7.6, 5.0))

# the gap the model must close, shaded between the two reference lines
ax.axhspan(floor, uniform, color=pal[7], alpha=0.08, zorder=0)
ax.axhline(uniform, ls="--", lw=1.3, color=pal[7],
           label=f"uniform baseline ({uniform:.3f})")
ax.axhline(floor, ls="--", lw=1.3, color=pal[2],
           label=f"Bayes-optimal floor ({floor:.3f})")

ax.plot(steps, train_ce, lw=1.0, color=pal[0], alpha=0.35, label="train minibatch CE")
ax.plot(steps, test_ce, lw=2.2, color=pal[0], label="held-out test CE")
ax.scatter([steps[-1]], [final], s=42, color=pal[0], zorder=5)

ax.set_xlabel("training step")
ax.set_ylabel("cross-entropy (nats / token)")
ax.set_ylim(floor - 0.12, uniform + 0.08)
ax.set_title("The model converges to the true RHM predictor")
ax.annotate(f"final test CE {final:.3f}\ncloses {gap_closed * 100:.0f}% of the gap",
            xy=(steps[-1], final), xytext=(steps[-1] * 0.5, final + 0.42),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="0.3"))
ax.legend(loc="upper right", framealpha=0.95)

plt.tight_layout()
out = os.path.join(OUTDIR, "loss_curve.png")
plt.savefig(out)
plt.close()
sz = os.path.getsize(out)
print("wrote", out, sz, "bytes")
assert sz > 10_000, f"{out} too small ({sz} bytes)"
