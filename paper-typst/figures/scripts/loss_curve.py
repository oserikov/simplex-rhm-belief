# Figure: loss_curve.png
# Message: The transformer learned the true predictor: its test cross-entropy
#   sits far below the uniform baseline and close to the Bayes-optimal floor.
# Data: artifacts/probe_data.npz: final_test_loss (model), uniform_baseline
#   (ln 8 = 2.079), floor (per-position Bayes-optimal entropy; mean over the 7
#   predicted positions = Bayes floor).
# Type: bar chart of three cross-entropies (nats/token) with reference lines.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _style import OUTDIR, PROBE, apply_style

apply_style()
d = np.load(PROBE)
model = float(d["final_test_loss"])
uniform = float(d["uniform_baseline"])
floor = float(np.mean(d["floor"]))     # Bayes-optimal over the 7 predicted positions
print("model", model, "uniform", uniform, "bayes_floor", floor)
assert np.isfinite([model, uniform, floor]).all(), "non-finite loss values"

names = ["Uniform\nbaseline", "Trained\ntransformer", "Bayes-optimal\nfloor"]
vals = [uniform, model, floor]
pal = sns.color_palette("colorblind")
colors = [pal[7], pal[0], pal[2]]

fig, ax = plt.subplots(figsize=(7.5, 5.2))
bars = ax.bar(range(3), vals, color=colors, edgecolor="0.2", width=0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(names)
ax.set_ylabel("Cross-entropy (nats / token)")
ax.set_ylim(0, uniform * 1.12)
ax.set_title("The model learned the true RHM predictor")
for i, v in enumerate(vals):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=11)
gap = (model - floor) / (uniform - floor) * 100
ax.annotate(f"closes {100 - gap:.0f}% of the\nuniform→Bayes gap",
            xy=(1, model), xytext=(1.35, uniform * 0.7), fontsize=10,
            ha="left", arrowprops=dict(arrowstyle="->", color="0.3"))

plt.tight_layout()
out = os.path.join(OUTDIR, "loss_curve.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")
