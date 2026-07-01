# Figures + table for the grammar generalization sweep (pass 2).
# Message: the pass-1 belief-geometry findings (inverted level gradient,
#   blooming, layer accumulation, causal steering) replicate across a family of
#   10 random grammars x 3 replicate seeds -- shown as distributions, not points.
# Data: globs results/sweep/g*/{config.json,analysis.json}.
# Emits into paper-typst/figures/:
#   sweep_level_r2.png   - violin/box of per-tree-level probe R^2 across runs
#   sweep_blooming.png   - overlaid radius-vs-k and entropy-vs-k, one line/grammar
#   sweep_steering.png   - wrong-latent log-p collapse: curves + spread
#   sweep_sanity.csv     - per-run sanity table for the appendix (typst reads it)
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from _style import OUTDIR, ROOT, apply_style

apply_style()

SWEEP_DIRS = sorted(glob.glob(os.path.join(ROOT, "results/sweep/g*")))
assert SWEEP_DIRS, "no results/sweep/g* runs found"

LEVELS = {
    "L0 (root)": ["root_L0"],
    "L1 (mid)": ["mid_L1a", "mid_L1b"],
    "L2 (leaf-parent)": ["low_L2a", "low_L2b", "low_L2c", "low_L2d"],
}


def load_runs():
    rows = []
    for d in SWEEP_DIRS:
        cfg_p, an_p = os.path.join(d, "config.json"), os.path.join(d, "analysis.json")
        if not (os.path.exists(cfg_p) and os.path.exists(an_p)):
            continue
        with open(cfg_p) as f:
            manifest = json.load(f)
        with open(an_p) as f:
            A = json.load(f)
        cfg = manifest["config"]
        rows.append({"cfg": cfg, "A": A, "name": os.path.basename(d)})
    assert rows, "no complete runs (need config.json + analysis.json)"
    return rows


RUNS = load_runs()
print(f"loaded {len(RUNS)} complete runs")


# ---- 1. per-level R^2 distribution (the inverted gradient as a distribution)
def fig_level_r2():
    recs = []
    for r in RUNS:
        lv = r["A"]["latent_level_r2"]
        for label, keys in LEVELS.items():
            recs.append({"level": label, "R2": float(np.mean([lv[k] for k in keys])),
                         "grammar": r["cfg"]["grammar"]})
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    order = list(LEVELS)
    sns.violinplot(data=df, x="level", y="R2", order=order, ax=ax,
                   inner=None, cut=0, color="0.85", linewidth=1)
    sns.stripplot(data=df, x="level", y="R2", order=order, ax=ax,
                  color=sns.color_palette("colorblind")[0], size=4, alpha=0.7, jitter=0.18)
    means = df.groupby("level")["R2"].mean().reindex(order)
    ax.plot(range(len(order)), means.values, "D", color=sns.color_palette("colorblind")[3],
            markersize=9, label="mean")
    ax.set_xlabel("Tree level of the decoded latent")
    ax.set_ylabel("Linear probe R²")
    ax.set_title("Inverted strength gradient replicates across grammars\n"
                 f"({len(RUNS)} runs = 10 grammars × 3 seeds)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTDIR, "sweep_level_r2.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print("  level means:", {k: round(v, 3) for k, v in means.items()})


# ---- 1b. per-level RMSE distribution (same as above, RMSE-valued) ----------
def fig_level_rmse():
    recs = []
    for r in RUNS:
        lv = r["A"]["latent_level_rmse"]
        for label, keys in LEVELS.items():
            recs.append({"level": label, "RMSE": float(np.mean([lv[k] for k in keys])),
                         "grammar": r["cfg"]["grammar"]})
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    order = list(LEVELS)
    sns.violinplot(data=df, x="level", y="RMSE", order=order, ax=ax,
                   inner=None, cut=0, color="0.85", linewidth=1)
    sns.stripplot(data=df, x="level", y="RMSE", order=order, ax=ax,
                  color=sns.color_palette("colorblind")[0], size=4, alpha=0.7, jitter=0.18)
    means = df.groupby("level")["RMSE"].mean().reindex(order)
    ax.plot(range(len(order)), means.values, "D", color=sns.color_palette("colorblind")[3],
            markersize=9, label="mean")
    ax.set_xlabel("Tree level of the decoded latent")
    ax.set_ylabel("Linear probe RMSE")
    ax.set_title("Inverted strength gradient replicates across grammars (RMSE)\n"
                 f"({len(RUNS)} runs = 10 grammars × 3 seeds)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTDIR, "sweep_level_rmse.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print("  level means:", {k: round(v, 3) for k, v in means.items()})


# ---- 2. overlaid blooming curves (radius + entropy vs context position) ----
def fig_blooming():
    by_g = {}
    for r in RUNS:
        g = r["cfg"]["grammar"]
        by_g.setdefault(g, {"rad": [], "ent": []})
        by_g[g]["rad"].append(r["A"]["blooming_radius_by_position"])
        by_g[g]["ent"].append(r["A"]["mean_posterior_entropy_by_position"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    cmap = sns.color_palette("viridis", len(by_g))
    for c, (g, dd) in zip(cmap, sorted(by_g.items()), strict=False):
        rad = np.mean(dd["rad"], 0); ent = np.mean(dd["ent"], 0)
        k = np.arange(1, len(rad) + 1)
        ax[0].plot(k, rad, "-o", color=c, ms=3, alpha=0.8, label=f"g{g}")
        ax[1].plot(k, ent, "-o", color=c, ms=3, alpha=0.8)
    ax[0].set_xlabel("context position k"); ax[0].set_ylabel("mean readout radius")
    ax[0].set_title("Residual representation blooms with context")
    ax[1].set_xlabel("context position k"); ax[1].set_ylabel("exact posterior entropy (nats)")
    ax[1].set_title("Belief sharpens with context")
    ax[0].legend(ncol=2, fontsize=8, title="grammar")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "sweep_blooming.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")


# ---- 3. steering: wrong-latent log-p collapse, curves + spread -------------
def fig_steering():
    curves, collapse = [], []
    alphas = None
    for r in RUNS:
        s = r["A"]["steering"]
        alphas = s["alphas"]
        w = np.array(s["true_lp_steer_wrong"])
        curves.append(w)
        collapse.append(float(w[0] - w[-1]))  # drop in log p(true) at max alpha
    curves = np.array(curves)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    for c in curves:
        ax[0].plot(alphas, c, "-", color="0.7", lw=0.8, alpha=0.6)
    ax[0].plot(alphas, curves.mean(0), "-o", color=sns.color_palette("colorblind")[3],
               lw=2.5, label="mean")
    ax[0].set_xlabel("patch strength α (steer → WRONG latent)")
    ax[0].set_ylabel("mean log p(true next token)")
    ax[0].set_title("Corrupting the belief breaks prediction\n(every run)")
    ax[0].legend()
    # match sweep_level_r2 style: grey violin body, jittered points, orange diamond mean
    cb = sns.color_palette("colorblind")
    sns.violinplot(y=collapse, ax=ax[1], inner=None, cut=0, color="0.85", linewidth=1)
    sns.stripplot(y=collapse, ax=ax[1], color=cb[0], size=5, alpha=0.75, jitter=0.08)
    ax[1].plot(0, np.mean(collapse), "D", color=cb[3], markersize=10, label="mean")
    ax[1].legend()
    pad = 0.2 * (np.max(collapse) - np.min(collapse) + 1e-9)
    ax[1].set_ylim(np.min(collapse) - pad, np.max(collapse) + pad)
    ax[1].set_ylabel(f"log p(true) collapse  (α=0 → α={alphas[-1]:.0f})")
    ax[1].set_title("Steering effect size across grammars")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "sweep_steering.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print(f"  collapse mean={np.mean(collapse):.2f} ± {np.std(collapse):.2f} nats")


# ---- 4. per-run sanity table (CSV, read by the typst appendix) -------------
def table_sanity():
    rows = []
    for r in RUNS:
        c, A = r["cfg"], r["A"]
        s = A["sanity"]
        rows.append({
            "grammar": c["grammar"], "seed": c["seed"],
            "n_layer": c["n_layer"], "n_embd": c["n_embd"], "n_head": c["n_head"],
            "test_ce": round(s["test_ce"], 3),
            "loss_gap_closed": round(s["loss_gap_closed"], 3),
            "root_r2": round(s["root_r2"], 3),
            "deepest_r2": round(s["deepest_r2"], 3),
            "root_rmse": round(s["root_rmse"], 3),
            "deepest_rmse": round(s["deepest_rmse"], 3),
        })
    df = pd.DataFrame(rows).sort_values(["grammar", "seed"]).reset_index(drop=True)
    out = os.path.join(OUTDIR, "sweep_sanity.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")
    # console summary of the headline distributions
    for lvl in ("root_r2", "deepest_r2", "loss_gap_closed"):
        print(f"  {lvl}: mean={df[lvl].mean():.3f} sd={df[lvl].std():.3f} "
              f"min={df[lvl].min():.3f} max={df[lvl].max():.3f}")
    return df


if __name__ == "__main__":
    fig_level_r2()
    fig_level_rmse()
    fig_blooming()
    fig_steering()
    table_sanity()
