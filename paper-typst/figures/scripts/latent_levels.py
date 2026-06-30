# Figure: latent_levels.png
# Message: The residual stream encodes the WHOLE latent hierarchy. Deeper /
#   more local latents are generally more linearly decodable than the global
#   root, with node-level variation across the four level-2 latents.
# Data: results/analysis.json: latent_level_r2 (root_L0, mid_L1a, mid_L1b,
#   low_L2a, low_L2b, low_L2c, low_L2d).
# Type: bar chart, R^2 per latent, ordered root -> deepest; root bar highlighted
#   as the spec's primary target.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import seaborn as sns
from _style import ANALYSIS, OUTDIR, apply_style

apply_style()
with open(ANALYSIS) as _f:
    A = json.load(_f)
lv = A["latent_level_r2"]
order = ["root_L0", "mid_L1a", "mid_L1b", "low_L2a", "low_L2b", "low_L2c", "low_L2d"]
labels = [
    "Root\n(L0)",
    "Mid\n(L1a)",
    "Mid\n(L1b)",
    "Low\n(L2a)",
    "Low\n(L2b)",
    "Low\n(L2c)",
    "Low\n(L2d)",
]
vals = [lv[k] for k in order]
assert len(vals) > 0, "Empty latent_level_r2"
print("latent_level_r2", list(zip(order, vals, strict=True)))

fig, ax = plt.subplots(figsize=(9.5, 5.2))
base = sns.color_palette("colorblind")
colors = [base[3]] + [base[0]] * (len(vals) - 1)   # highlight root
bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="0.2", width=0.7)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels)
ax.set_ylabel("Linear probe R²")
ax.set_ylim(0, 0.75)
ax.set_xlabel("Latent variable (global root → local leaf-level)")
ax.set_title("Local latents are generally more strongly encoded")
for i, v in enumerate(vals):
    ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=10)
plt.tight_layout()
out = os.path.join(OUTDIR, "latent_levels.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")
