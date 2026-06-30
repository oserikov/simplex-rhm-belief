# Figure: ruletable_graph.png
# Message: the frozen rule table of @ruletable, drawn as a 4-column DAG
#   (root -> level 1 -> level 2 -> leaves) so the "no cross-level ambiguity,
#   each level points only to the level below it" structure is visible at a
#   glance, instead of read off three separate text columns.
# Data: artifacts/grammar.npz: rules_0, rules_1, rules_2 (each (v, m, s)).
import colorsys
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from _style import OUTDIR, ROOT
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

d = np.load(os.path.join(ROOT, "artifacts/grammar.npz"))
v = int(d["v"])  # 8
levels = [d["rules_0"], d["rules_1"], d["rules_2"]]  # each (v, m, s)
col_titles = ["Root", "Level 1", "Level 2", "Leaves"]

fig, ax = plt.subplots(figsize=(15, 11))
x_cols = [0.0, 1.4, 2.8, 4.2]
y_scale = 1.6
y_pos = {i: (v - 1 - i) * y_scale for i in range(v)}  # top->bottom S1..S8

node_xy = {}
for c, x in enumerate(x_cols):
    for s in range(v):
        y = y_pos[s]
        node_xy[(c, s)] = (x, y)
        ax.scatter([x], [y], s=900, facecolors="white", edgecolors="black",
                   zorder=5, linewidths=1.3)
        ax.text(x, y, f"S{s + 1}", ha="center", va="center", fontsize=11, zorder=6)
    ax.text(x, (v - 1) * y_scale + 1.0, col_titles[c], ha="center", va="bottom",
            fontsize=13, fontweight="bold")

# 16 distinct hues (one per node x rule), stepped by the golden angle so that
# *every* pair of indices -- same node's two rules, or neighboring nodes -- lands
# far apart on the hue wheel. (A "paired" palette like tab20 puts same-node rules
# right next to each other in hue, which is exactly what we don't want.)
GOLDEN = 0.6180339887498949
node_rule_color = {}
for p in range(v):
    for ri in range(2):
        i = p * 2 + ri
        hue = (i * GOLDEN) % 1.0
        sat = 0.75 if ri == 0 else 0.55
        val = 0.85
        node_rule_color[(p, ri)] = colorsys.hsv_to_rgb(hue, sat, val)

# For each transition, draw edges with per-(parent,rule,child-slot) curvature
# chosen so that parallel edges between the same column-pair fan out and don't
# coincide. Curvature depends on the signed vertical gap AND a small per-edge
# offset keyed by source row, so edges from different sources never overlap.
for c in range(3):
    rules = levels[c]  # (v, m, s)
    for parent in range(v):
        for ri in range(rules.shape[1]):
            color = node_rule_color[(parent, ri)]
            for j in range(rules.shape[2]):
                child = int(rules[parent, ri, j])
                x0, y0 = node_xy[(c, parent)]
                x1, y1 = node_xy[(c + 1, child)]
                gap = y1 - y0
                edge_idx = ri * rules.shape[2] + j  # 0..3, unique per (parent, rule, slot)
                base = 0.04 + 0.006 * abs(gap)
                sign = 1 if (parent + edge_idx) % 2 == 0 else -1
                jitter = 0.015 * (edge_idx - 1.5) + 0.005 * ((parent * 7) % 9 - 4)
                rad = sign * base + jitter
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>", mutation_scale=10,
                    color=color, lw=1.0, alpha=0.55, zorder=2,
                    shrinkA=15, shrinkB=15,
                )
                ax.add_patch(arrow)

ax.set_xlim(-0.6, 4.8)
ax.set_ylim(-1.5, (v - 1) * y_scale + 1.8)
ax.axis("off")
fig.suptitle("RHM rule table as a graph (root -> level 1 -> level 2 -> leaves)",
             fontsize=14, y=0.99)

handles = [Line2D([0], [0], color=node_rule_color[(p, ri)], lw=2,
                  label=f"S{p + 1} rule {ri + 1}")
           for p in range(v) for ri in range(2)]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
           ncol=8, fontsize=8, frameon=False, columnspacing=1.0, handlelength=1.5)

plt.tight_layout(rect=(0, 0, 1, 0.90))
out = os.path.join(OUTDIR, "ruletable_graph.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("wrote", out, os.path.getsize(out), "bytes")
