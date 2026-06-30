# Figures + tables for the depth-4 grammar sweep (pass 4).
# Message: repeat the pass-1/2 belief-geometry replication one tree level deeper
#   (L=4, context d=16, four latent levels L0..L3) across 10 grammars x 3 seeds
#   x n_layer{2,3} = 60 runs, and score the three pre-registered L=4 predictions.
# Data: globs results/depth4/g*/{config.json,analysis.json}.
# Emits into paper-typst/figures/:
#   depth4_level_r2.png    - violin/strip of per-tree-level probe R^2 (four levels)
#   depth4_blooming.png    - root readout radius + exact root entropy vs the 16
#                            context positions (scores "does the root bloom?")
#   depth4_layer_r2.png    - root probe R^2 vs layer, one line per n_layer in {2,3}
#   depth4_steering.png    - wrong-latent log-p collapse: curves + spread
#   depth4_sanity.csv      - per-run sanity table for the appendix (typst reads it)
#   depth4_prereg.json     - scored-prediction numbers for the scorecard
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

DEPTH4_DIRS = sorted(glob.glob(os.path.join(ROOT, "results/depth4/g*")))
assert DEPTH4_DIRS, "no results/depth4/g* runs found"

# Four latent levels of the L=4 tree (keys emitted by analyze.py for g.L==4).
LEVELS = {
    "L0 (root)": ["root_L0"],
    "L1": ["L1_0", "L1_1"],
    "L2": ["L2_0", "L2_1", "L2_2", "L2_3"],
    "L3 (leaf-parent)": [f"L3_{p}" for p in range(8)],
}
CB = sns.color_palette("colorblind")


def load_runs(dirs):
    rows = []
    for d in dirs:
        cfg_p, an_p = os.path.join(d, "config.json"), os.path.join(d, "analysis.json")
        if not (os.path.exists(cfg_p) and os.path.exists(an_p)):
            continue
        with open(cfg_p) as f:
            manifest = json.load(f)
        with open(an_p) as f:
            A = json.load(f)
        rows.append({"cfg": manifest["config"], "A": A, "name": os.path.basename(d)})
    return rows


RUNS = load_runs(DEPTH4_DIRS)
assert RUNS, "no complete depth4 runs (need config.json + analysis.json)"
print(f"loaded {len(RUNS)} complete depth4 runs")


def level_means_per_run(r):
    lv = r["A"]["latent_level_r2"]
    return {label: float(np.mean([lv[k] for k in keys])) for label, keys in LEVELS.items()}


# ---- 1. per-level R^2 distribution across the four levels -------------------
def fig_level_r2():
    recs = []
    for r in RUNS:
        for label, val in level_means_per_run(r).items():
            recs.append({"level": label, "R2": val,
                         "grammar": r["cfg"]["grammar"], "n_layer": r["cfg"]["n_layer"]})
    df = pd.DataFrame(recs)
    order = list(LEVELS)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    sns.violinplot(data=df, x="level", y="R2", order=order, ax=ax,
                   inner=None, cut=0, color="0.85", linewidth=1)
    sns.stripplot(data=df, x="level", y="R2", order=order, ax=ax,
                  color=CB[0], size=4, alpha=0.7, jitter=0.18)
    means = df.groupby("level")["R2"].mean().reindex(order)
    ax.plot(range(len(order)), means.values, "D", color=CB[3], markersize=9, label="mean")
    ax.set_xlabel("Tree level of the decoded latent (root → leaf-parent)")
    ax.set_ylabel("Linear probe R²")
    ax.set_title("L=4 per-level probe strength\n"
                 f"({len(RUNS)} runs = 10 grammars × 3 seeds × n_layer∈{{2,3}})")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTDIR, "depth4_level_r2.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print("  level means:", {k: round(v, 3) for k, v in means.items()})
    return means


# ---- 2. root blooming over the 16-position window --------------------------
def fig_blooming():
    rad = np.array([r["A"]["blooming_radius_by_position"] for r in RUNS])
    ent = np.array([r["A"]["mean_posterior_entropy_by_position"] for r in RUNS])
    k = np.arange(1, rad.shape[1] + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for c in rad:
        ax[0].plot(k, c, "-", color="0.8", lw=0.6, alpha=0.5)
    ax[0].plot(k, rad.mean(0), "-o", color=CB[3], lw=2.5, ms=4, label="mean")
    ax[0].set_xlabel("context position k"); ax[0].set_ylabel("root readout radius")
    ax[0].set_title("Does the root belief bloom? (readout radius)")
    ax[0].legend()
    for c in ent:
        ax[1].plot(k, c, "-", color="0.8", lw=0.6, alpha=0.5)
    ax[1].plot(k, ent.mean(0), "-o", color=CB[0], lw=2.5, ms=4, label="mean")
    ax[1].set_xlabel("context position k"); ax[1].set_ylabel("exact root posterior entropy (nats)")
    ax[1].set_title("Exact root posterior entropy vs context")
    ax[1].legend()
    plt.tight_layout()
    out = os.path.join(OUTDIR, "depth4_blooming.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    return rad.mean(0), ent.mean(0)


# ---- 3. layer-wise root-belief accumulation, split by n_layer --------------
def fig_layer_r2():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    summary = {}
    for nl, color in zip((2, 3), (CB[0], CB[3]), strict=False):
        curves = [r["A"]["layer_r2"] for r in RUNS if r["cfg"]["n_layer"] == nl]
        if not curves:
            continue
        curves = np.array(curves)
        x = np.arange(curves.shape[1])
        m, sd = curves.mean(0), curves.std(0)
        ax.plot(x, m, "-o", color=color, lw=2, label=f"n_layer={nl}  ({len(curves)} runs)")
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15)
        summary[nl] = m.tolist()
    ax.set_xlabel("layer (0 = token+pos embedding)")
    ax.set_ylabel("root-belief probe R²")
    ax.set_title("Belief accumulates across layers at L=4")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUTDIR, "depth4_layer_r2.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    return summary


# ---- 4. steering: wrong-latent log-p collapse ------------------------------
def fig_steering():
    curves, collapse, alphas = [], [], None
    for r in RUNS:
        s = r["A"]["steering"]
        alphas = s["alphas"]
        w = np.array(s["true_lp_steer_wrong"])
        curves.append(w)
        collapse.append(float(w[0] - w[-1]))
    curves = np.array(curves)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    for c in curves:
        ax[0].plot(alphas, c, "-", color="0.7", lw=0.7, alpha=0.5)
    ax[0].plot(alphas, curves.mean(0), "-o", color=CB[3], lw=2.5, label="mean")
    ax[0].set_xlabel("patch strength α (steer → WRONG latent)")
    ax[0].set_ylabel("mean log p(true next token)")
    ax[0].set_title("Corrupting the belief breaks prediction\n(every L=4 run)")
    ax[0].legend()
    # match sweep_level_r2 style: grey violin body, jittered points, orange diamond mean
    sns.violinplot(y=collapse, ax=ax[1], inner=None, cut=0, color="0.85", linewidth=1)
    sns.stripplot(y=collapse, ax=ax[1], color=CB[0], size=5, alpha=0.75, jitter=0.08)
    ax[1].plot(0, np.mean(collapse), "D", color=CB[3], markersize=10, label="mean")
    ax[1].legend()
    pad = 0.2 * (np.max(collapse) - np.min(collapse) + 1e-9)
    ax[1].set_ylim(np.min(collapse) - pad, np.max(collapse) + pad)
    ax[1].set_ylabel(f"log p(true) collapse  (α=0 → α={alphas[-1]:.0f})")
    ax[1].set_title("Steering effect size across the L=4 family")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "depth4_steering.png")
    plt.savefig(out); plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print(f"  collapse mean={np.mean(collapse):.2f} ± {np.std(collapse):.2f} nats")
    return float(np.mean(collapse)), float(np.std(collapse))


# ---- 5. per-run sanity table -----------------------------------------------
def table_sanity():
    rows = []
    for r in RUNS:
        c, s = r["cfg"], r["A"]["sanity"]
        rows.append({
            "grammar": c["grammar"], "seed": c["seed"], "n_layer": c["n_layer"],
            "test_ce": round(s["test_ce"], 3),
            "loss_gap_closed": round(s["loss_gap_closed"], 3),
            "root_r2": round(s["root_r2"], 3),
            "deepest_r2": round(s["deepest_r2"], 3),
        })
    df = pd.DataFrame(rows).sort_values(["grammar", "seed", "n_layer"]).reset_index(drop=True)
    out = os.path.join(OUTDIR, "depth4_sanity.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")
    return df


# ---- L=3 pass-2 n_layer=2 root mean for the arch-controlled comparison ------
def l3_n2_root_mean():
    dirs = sorted(glob.glob(os.path.join(ROOT, "results/sweep/g*")))
    runs = load_runs(dirs)
    vals = [r["A"]["latent_level_r2"]["root_L0"] for r in runs
            if r["cfg"].get("n_layer") == 2]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


if __name__ == "__main__":
    means = fig_level_r2()
    rad_mean, ent_mean = fig_blooming()
    layer_summary = fig_layer_r2()
    collapse_mean, collapse_sd = fig_steering()
    df = table_sanity()

    # per-run monotone-ladder check (prediction 3): L0 < L1 < L2 < L3 strictly
    ladder_ok = 0
    for r in RUNS:
        lm = level_means_per_run(r)
        seq = [lm["L0 (root)"], lm["L1"], lm["L2"], lm["L3 (leaf-parent)"]]
        ladder_ok += all(seq[i] < seq[i + 1] for i in range(3))

    l3_root, l3_n = l3_n2_root_mean()
    root_n2 = df[df["n_layer"] == 2]["root_r2"]
    root_n3 = df[df["n_layer"] == 3]["root_r2"]

    prereg = {
        "n_runs": len(RUNS),
        "level_means": {k: round(float(v), 4) for k, v in means.items()},
        "root_r2_mean_all": round(float(df["root_r2"].mean()), 4),
        "root_r2_mean_n2": round(float(root_n2.mean()), 4),
        "root_r2_mean_n3": round(float(root_n3.mean()), 4),
        "l3_pass2_n2_root_mean": (round(l3_root, 4) if l3_root is not None else None),
        "l3_pass2_n2_count": l3_n,
        "blooming_root_radius_first": round(float(rad_mean[0]), 4),
        "blooming_root_radius_last": round(float(rad_mean[-1]), 4),
        "blooming_root_entropy_first": round(float(ent_mean[0]), 4),
        "blooming_root_entropy_last": round(float(ent_mean[-1]), 4),
        "ladder_monotone_fraction": round(ladder_ok / len(RUNS), 4),
        "ladder_monotone_count": f"{ladder_ok}/{len(RUNS)}",
        "steering_collapse_mean": round(collapse_mean, 4),
        "steering_collapse_sd": round(collapse_sd, 4),
    }
    out = os.path.join(OUTDIR, "depth4_prereg.json")
    with open(out, "w") as f:
        json.dump(prereg, f, indent=2)
    print("wrote", out)
    print(json.dumps(prereg, indent=2))
