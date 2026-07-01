# Figure: layer_position.png
# Message: Belief decodability ACCUMULATES across layers and BLOOMS across
#   context positions. (a) Probe R^2 for the root posterior rises layer by layer
#   while a shuffled-label control stays at ~0. (b) Heatmap of R^2(layer,
#   position): early positions are decodable even at shallow layers; the final
#   layer pushes decodability deep into the sequence.
# Data: results/analysis.json: layer_r2, layer_r2_shuffled, heatmap_layer_position_r2.
# Type: 2 panels. (a) grouped line/marker with baseline; (b) annotated heatmap.
# Note: a few heatmap cells are large negatives (probe far worse than mean).
#   Colors are clipped to [-1, 1] for legibility, but annotations show raw values.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _style import ANALYSIS, OUTDIR, apply_style

apply_style()
with open(ANALYSIS) as _f:
    A = json.load(_f)
layer = np.array(A["layer_r2"])
shuf = np.array(A["layer_r2_shuffled"])
heat = np.array(A["heatmap_layer_position_r2"])
nL, nP = heat.shape
assert layer.size == nL, "layer dims mismatch"
print("layer_r2", layer, "shuffled", shuf, "heat", heat.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.25]})
pal = sns.color_palette("colorblind")
lx = np.arange(nL)
ax1.plot(lx, layer, "-o", color=pal[0], lw=2.2, ms=9, label="probe R²")
ax1.plot(lx, shuf, "--s", color=pal[3], lw=2.0, ms=7, label="shuffled-label control")
ax1.axhline(0, color="0.6", lw=1)
ax1.set_xticks(lx)
ax1.set_xticklabels([f"Layer {i}" for i in lx])
ax1.set_ylabel("Root posterior R²")
ax1.set_ylim(-0.15, 0.55)
ax1.set_title("Decodability accumulates across layers")
ax1.legend(loc="upper left")
for i, y in enumerate(layer):
    ax1.annotate(f"{y:.2f}", (lx[i], y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=9)

def fmt_r2(x):
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 100:
        return f"{x:.1e}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


heat_clip = np.clip(heat, -1, 1)
heat_annot = np.vectorize(fmt_r2)(heat)
hm = sns.heatmap(heat_clip, ax=ax2, cmap="viridis", vmin=-1, vmax=1,
                 annot=heat_annot, fmt="", annot_kws={"size": 8},
                 cbar_kws={"label": "R² color scale (clipped to [-1,1])"},
                 linewidths=0.5, linecolor="white")
ax2.set_xlabel("Context position $k$")
ax2.set_ylabel("Layer")
ax2.set_yticklabels([f"{i}" for i in range(nL)], rotation=0)
ax2.set_title("R²(layer, position)")

plt.tight_layout()
out = os.path.join(OUTDIR, "layer_position.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")

# ---- RMSE variant ------------------------------------------------------
layer_rmse = np.array(A["layer_rmse"])
shuf_rmse = np.array(A["layer_rmse_shuffled"])
heat_rmse = np.array(A["heatmap_layer_position_rmse"])
assert layer_rmse.size == nL, "layer dims mismatch"
print("layer_rmse", layer_rmse, "shuffled", shuf_rmse, "heat", heat_rmse.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.25]})
ax1.plot(lx, layer_rmse, "-o", color=pal[0], lw=2.2, ms=9, label="probe RMSE")
ax1.plot(lx, shuf_rmse, "--s", color=pal[3], lw=2.0, ms=7, label="shuffled-label control")
ax1.set_xticks(lx)
ax1.set_xticklabels([f"Layer {i}" for i in lx])
ax1.set_ylabel("Root posterior RMSE")
ax1.set_title("Decodability (RMSE) accumulates across layers")
ax1.legend(loc="upper left")
for i, y in enumerate(layer_rmse):
    ax1.annotate(f"{y:.2f}", (lx[i], y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=9)

heat_annot_rmse = np.vectorize(fmt_r2)(heat_rmse)
hm = sns.heatmap(heat_rmse, ax=ax2, cmap="viridis", vmin=0,
                 annot=heat_annot_rmse, fmt="", annot_kws={"size": 8},
                 cbar_kws={"label": "RMSE color scale"},
                 linewidths=0.5, linecolor="white")
ax2.set_xlabel("Context position $k$")
ax2.set_ylabel("Layer")
ax2.set_yticklabels([f"{i}" for i in range(nL)], rotation=0)
ax2.set_title("RMSE(layer, position)")

plt.tight_layout()
out = os.path.join(OUTDIR, "layer_position_rmse.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")
