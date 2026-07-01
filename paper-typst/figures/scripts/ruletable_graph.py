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
from _style import OUTDIR, ROOT
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def draw_grammar(g, out_path, show_weights=False, title=None):
    """Draw a Grammar (rhm.Grammar-like: .v, .rules, .probs) as a 4-column DAG.

    If show_weights, each edge is labeled near its start with the rule's
    choice probability and linewidth scales with that probability.
    """
    v = int(g.v)
    levels = g.rules  # list of (v, m, s) arrays, one per level
    probs = g.probs  # list of (v, m) arrays, one per level
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
        rule_probs = probs[c]  # (v, m)
        for parent in range(v):
            for ri in range(rules.shape[1]):
                color = node_rule_color[(parent, ri)]
                p_rule = float(rule_probs[parent, ri])
                lw = max(0.4, min(3.0, 0.4 + 2.6 * p_rule)) if show_weights else 1.0
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
                        color=color, lw=lw, alpha=0.55, zorder=2,
                        shrinkA=15, shrinkB=15,
                    )
                    ax.add_patch(arrow)
                    if show_weights:
                        xs, ys = x0 + 0.15 * (x1 - x0), y0 + 0.15 * (y1 - y0)
                        ax.text(xs, ys, f"{p_rule:.2f}", fontsize=6.5, color=color,
                                ha="center", va="center", zorder=7,
                                bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                          ec="none", alpha=0.7))

    ax.set_xlim(-0.6, 4.8)
    ax.set_ylim(-1.5, (v - 1) * y_scale + 1.8)
    ax.axis("off")
    fig.suptitle(title or "RHM rule table as a graph (root -> level 1 -> level 2 -> leaves)",
                 fontsize=14, y=0.99)

    handles = [Line2D([0], [0], color=node_rule_color[(p, ri)], lw=2,
                      label=f"S{p + 1} rule {ri + 1}")
               for p in range(v) for ri in range(2)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=8, fontsize=8, frameon=False, columnspacing=1.0, handlelength=1.5)

    plt.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path, os.path.getsize(out_path), "bytes")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    from analyze import load_grammar

    g_base = load_grammar(os.path.join(ROOT, "artifacts"))
    draw_grammar(g_base, os.path.join(OUTDIR, "ruletable_graph.png"))

    g_skew = load_grammar(
        os.path.join(ROOT, "results/noncollapse/skhigh_am0_g00_L2_d128_h4_t4000_s0"))
    draw_grammar(g_skew, os.path.join(OUTDIR, "ruletable_graph_skew.png"),
                 show_weights=True,
                 title="Skewed grammar (Dirichlet skew=high) as a graph")

    g_amb = load_grammar(
        os.path.join(ROOT, "results/noncollapse/sknone_am0.6_g00_L2_d128_h4_t4000_s0"))
    draw_grammar(g_amb, os.path.join(OUTDIR, "ruletable_graph_amb.png"),
                 show_weights=False,
                 title="Ambiguous grammar (rho=0.6 shared child-tuples) as a graph")
