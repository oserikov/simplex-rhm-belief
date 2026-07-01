"""Figures + table for the architecture-capacity sweep (pass 3).

Message: the pass-1/2 belief-geometry phenomenology (linear root/level
decodability, the inverted level gradient, layer-wise accumulation) is robust
across model **capacity** down to a floor; the root is information-limited while
the leaves are capacity/optimization-limited; and decodability is not merely a
restatement of how well the model fit the loss.

Data: globs results/arch/*/{config.json,analysis.json} -- the 11 OAT configs
(one-axis-at-a-time around the pass-1 baseline L2 d128 h4 t4000) x grammar{0,1,2}
x seed{0,1,2} = 99 runs.

Emits into paper-typst/figures/:
  arch_marginal_r2.png   - 2x2: root/mid/leaf-parent R^2 vs each axis
                           (n_embd, n_layer, n_head, steps), error bars over
                           grammars x seeds, other three axes held at baseline.
  arch_depth_accum.png   - layer-wise root R^2 vs normalized depth, one line per
                           n_layer; + final-layer root R^2 vs n_layer.
  arch_lossfit.png       - root R^2 and leaf R^2 vs loss_gap_closed, all runs.
  arch_sanity.csv        - per-run sanity table for the appendix (typst reads it).
"""
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

ARCH_DIRS = sorted(glob.glob(os.path.join(ROOT, "results/arch/g*")))
assert ARCH_DIRS, "no results/arch/g* runs found"

# Published pass-1 baseline = shared center of all four axis slices.
BASE = dict(n_layer=2, n_embd=128, n_head=4, steps=4000)

# Tree levels (RHM s=2, L=3): root L0, two mids L1, four leaf-parents L2.
LEVELS = {
    "L0 (root)": ["root_L0"],
    "L1 (mid)": ["mid_L1a", "mid_L1b"],
    "L2 (leaf-parent)": ["low_L2a", "low_L2b", "low_L2c", "low_L2d"],
}
AXES = ["n_embd", "n_layer", "n_head", "steps"]
AXIS_LABEL = {
    "n_embd": "width  (n_embd)",
    "n_layer": "depth  (n_layer)",
    "n_head": "heads  (n_head)",
    "steps": "budget  (training steps)",
}
CB = sns.color_palette("colorblind")
LEVEL_COLOR = {"L0 (root)": CB[3], "L1 (mid)": CB[0], "L2 (leaf-parent)": CB[2]}


def load_runs():
    rows = []
    for d in ARCH_DIRS:
        cfg_p = os.path.join(d, "config.json")
        an_p = os.path.join(d, "analysis.json")
        if not (os.path.exists(cfg_p) and os.path.exists(an_p)):
            continue
        with open(cfg_p) as f:
            cfg = json.load(f)["config"]
        with open(an_p) as f:
            A = json.load(f)
        rows.append({"cfg": cfg, "A": A, "name": os.path.basename(d)})
    assert rows, "no complete runs (need config.json + analysis.json)"
    return rows


RUNS = load_runs()
print(f"loaded {len(RUNS)} complete arch runs")


def level_value(A, level):
    lv = A["latent_level_r2"]
    return float(np.mean([lv[k] for k in LEVELS[level]]))


def level_value_rmse(A, level):
    lv = A["latent_level_rmse"]
    return float(np.mean([lv[k] for k in LEVELS[level]]))


def on_axis_slice(axis):
    """Runs that vary `axis` while the other three axes sit at baseline."""
    others = [a for a in AXES if a != axis]
    out = []
    for r in RUNS:
        c = r["cfg"]
        if all(c[o] == BASE[o] for o in others):
            out.append(r)
    return out


# ---- 1. marginal-effect curves: 2x2, root/mid/leaf R^2 vs each axis ---------
def fig_marginal():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    summary = {}
    for ax, axis in zip(axes.flat, AXES, strict=False):
        slice_runs = on_axis_slice(axis)
        xs = sorted({r["cfg"][axis] for r in slice_runs})
        for level in LEVELS:
            means, errs = [], []
            for x in xs:
                vals = [level_value(r["A"], level)
                        for r in slice_runs if r["cfg"][axis] == x]
                means.append(np.mean(vals))
                errs.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
            ax.errorbar(range(len(xs)), means, yerr=errs, marker="o",
                        capsize=3, lw=2, color=LEVEL_COLOR[level], label=level)
            summary[(axis, level)] = list(
                zip(xs, [round(m, 3) for m in means], strict=False))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(AXIS_LABEL[axis])
        ax.set_ylabel("Linear probe R²")
        ax.set_ylim(-0.02, None)
        ax.axvline(xs.index(BASE[axis]), color="0.6", ls=":", lw=1, zorder=0)
        ax.set_title(f"R² vs {AXIS_LABEL[axis]}")
    axes.flat[0].legend(title="tree level", loc="lower right")
    fig.suptitle("Marginal capacity effects on belief decodability "
                 "(OAT around L2·d128·h4·t4000; error bars = SEM over "
                 "3 grammars × 3 seeds)", y=1.00)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "arch_marginal_r2.png")
    plt.savefig(out)
    plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    for k, v in summary.items():
        print("  ", k, v)


# ---- 1b. marginal-effect curves, RMSE-valued --------------------------------
def fig_marginal_rmse():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    summary = {}
    for ax, axis in zip(axes.flat, AXES, strict=False):
        slice_runs = on_axis_slice(axis)
        xs = sorted({r["cfg"][axis] for r in slice_runs})
        for level in LEVELS:
            means, errs = [], []
            for x in xs:
                vals = [level_value_rmse(r["A"], level)
                        for r in slice_runs if r["cfg"][axis] == x]
                means.append(np.mean(vals))
                errs.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
            ax.errorbar(range(len(xs)), means, yerr=errs, marker="o",
                        capsize=3, lw=2, color=LEVEL_COLOR[level], label=level)
            summary[(axis, level)] = list(
                zip(xs, [round(m, 3) for m in means], strict=False))
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(AXIS_LABEL[axis])
        ax.set_ylabel("Linear probe RMSE")
        ax.axvline(xs.index(BASE[axis]), color="0.6", ls=":", lw=1, zorder=0)
        ax.set_title(f"RMSE vs {AXIS_LABEL[axis]}")
    axes.flat[0].legend(title="tree level", loc="upper right")
    fig.suptitle("Marginal capacity effects on belief decodability (RMSE)\n"
                 "(OAT around L2·d128·h4·t4000; error bars = SEM over "
                 "3 grammars × 3 seeds)", y=1.00)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "arch_marginal_rmse.png")
    plt.savefig(out)
    plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    for k, v in summary.items():
        print("  ", k, v)


# ---- 2. depth & accumulation -----------------------------------------------
def fig_depth_accum():
    depth_runs = on_axis_slice("n_layer")
    by_L = {}
    for r in depth_runs:
        by_L.setdefault(r["cfg"]["n_layer"], []).append(r["A"]["layer_r2"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cmap = sns.color_palette("viridis", len(by_L))
    final_by_L = {}
    for c, (L, curves) in zip(cmap, sorted(by_L.items()), strict=False):
        curves = np.array(curves)            # (reps, n_layer+1)
        mean = curves.mean(0)
        sem = curves.std(0) / np.sqrt(curves.shape[0])
        # residual index 0..n_layer -> normalized depth 0..1
        norm = np.arange(len(mean)) / (len(mean) - 1)
        ax[0].errorbar(norm, mean, yerr=sem, marker="o", color=c, lw=2,
                       capsize=2, label=f"n_layer={L}")
        final_by_L[L] = (mean[-1], sem[-1])
    ax[0].set_xlabel("normalized depth  (residual index / n_layer)")
    ax[0].set_ylabel("root-belief probe R²")
    ax[0].set_title("Belief accumulates across depth")
    ax[0].legend(title="depth")

    Ls = sorted(final_by_L)
    ax[1].errorbar(Ls, [final_by_L[L][0] for L in Ls],
                   yerr=[final_by_L[L][1] for L in Ls],
                   marker="s", color=CB[3], lw=2, capsize=3)
    ax[1].set_xticks(Ls)
    ax[1].set_xlabel("n_layer")
    ax[1].set_ylabel("final-layer root R²")
    ax[1].set_title("Deeper models lift the final readout")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "arch_depth_accum.png")
    plt.savefig(out)
    plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print("  final root R² by n_layer:",
          {L: round(final_by_L[L][0], 3) for L in Ls})


# ---- 2b. depth & accumulation, RMSE-valued ----------------------------------
def fig_depth_accum_rmse():
    depth_runs = on_axis_slice("n_layer")
    by_L = {}
    for r in depth_runs:
        by_L.setdefault(r["cfg"]["n_layer"], []).append(r["A"]["layer_rmse"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cmap = sns.color_palette("viridis", len(by_L))
    final_by_L = {}
    for c, (L, curves) in zip(cmap, sorted(by_L.items()), strict=False):
        curves = np.array(curves)            # (reps, n_layer+1)
        mean = curves.mean(0)
        sem = curves.std(0) / np.sqrt(curves.shape[0])
        norm = np.arange(len(mean)) / (len(mean) - 1)
        ax[0].errorbar(norm, mean, yerr=sem, marker="o", color=c, lw=2,
                       capsize=2, label=f"n_layer={L}")
        final_by_L[L] = (mean[-1], sem[-1])
    ax[0].set_xlabel("normalized depth  (residual index / n_layer)")
    ax[0].set_ylabel("root-belief probe RMSE")
    ax[0].set_title("Belief accumulates across depth (RMSE)")
    ax[0].legend(title="depth")

    Ls = sorted(final_by_L)
    ax[1].errorbar(Ls, [final_by_L[L][0] for L in Ls],
                   yerr=[final_by_L[L][1] for L in Ls],
                   marker="s", color=CB[3], lw=2, capsize=3)
    ax[1].set_xticks(Ls)
    ax[1].set_xlabel("n_layer")
    ax[1].set_ylabel("final-layer root RMSE")
    ax[1].set_title("Deeper models lift the final readout (RMSE)")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "arch_depth_accum_rmse.png")
    plt.savefig(out)
    plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")
    print("  final root RMSE by n_layer:",
          {L: round(final_by_L[L][0], 3) for L in Ls})


# ---- 3. loss-fit vs decodability scatter (all runs) ------------------------
def fig_lossfit():
    lgc, root, leaf = [], [], []
    for r in RUNS:
        lgc.append(r["A"]["sanity"]["loss_gap_closed"])
        root.append(level_value(r["A"], "L0 (root)"))
        leaf.append(level_value(r["A"], "L2 (leaf-parent)"))
    lgc, root, leaf = map(np.array, (lgc, root, leaf))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(lgc, root, s=34, alpha=0.75, color=LEVEL_COLOR["L0 (root)"],
               label="root (L0)", edgecolor="white", linewidth=0.4)
    ax.scatter(lgc, leaf, s=34, alpha=0.75, color=LEVEL_COLOR["L2 (leaf-parent)"],
               label="leaf-parent (L2)", edgecolor="white", linewidth=0.4)
    for y, col in ((root, LEVEL_COLOR["L0 (root)"]),
                   (leaf, LEVEL_COLOR["L2 (leaf-parent)"])):
        if len(np.unique(lgc)) > 1:
            b, a = np.polyfit(lgc, y, 1)
            xs = np.linspace(lgc.min(), lgc.max(), 50)
            ax.plot(xs, a + b * xs, color=col, lw=1.5, ls="--", alpha=0.8)
            r = np.corrcoef(lgc, y)[0, 1]
            print(f"  corr(loss_gap_closed, {'root' if col==LEVEL_COLOR['L0 (root)'] else 'leaf'})"
                  f" = {r:.3f}")
    ax.set_xlabel("loss gap closed  (CE: uniform → Bayes floor)")
    ax.set_ylabel("Linear probe R²")
    ax.set_title("Decodability vs loss fit across all 99 runs\n"
                 "(does belief geometry just track how well the model fit?)")
    ax.legend(title="tree level")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "arch_lossfit.png")
    plt.savefig(out)
    plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")


# ---- 3b. loss-fit vs decodability scatter, RMSE-valued ---------------------
def fig_lossfit_rmse():
    lgc, root, leaf = [], [], []
    for r in RUNS:
        lgc.append(r["A"]["sanity"]["loss_gap_closed"])
        root.append(level_value_rmse(r["A"], "L0 (root)"))
        leaf.append(level_value_rmse(r["A"], "L2 (leaf-parent)"))
    lgc, root, leaf = map(np.array, (lgc, root, leaf))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(lgc, root, s=34, alpha=0.75, color=LEVEL_COLOR["L0 (root)"],
               label="root (L0)", edgecolor="white", linewidth=0.4)
    ax.scatter(lgc, leaf, s=34, alpha=0.75, color=LEVEL_COLOR["L2 (leaf-parent)"],
               label="leaf-parent (L2)", edgecolor="white", linewidth=0.4)
    for y, col in ((root, LEVEL_COLOR["L0 (root)"]),
                   (leaf, LEVEL_COLOR["L2 (leaf-parent)"])):
        if len(np.unique(lgc)) > 1:
            b, a = np.polyfit(lgc, y, 1)
            xs = np.linspace(lgc.min(), lgc.max(), 50)
            ax.plot(xs, a + b * xs, color=col, lw=1.5, ls="--", alpha=0.8)
            r = np.corrcoef(lgc, y)[0, 1]
            print(f"  corr(loss_gap_closed, {'root' if col==LEVEL_COLOR['L0 (root)'] else 'leaf'})"
                  f" rmse = {r:.3f}")
    ax.set_xlabel("loss gap closed  (CE: uniform → Bayes floor)")
    ax.set_ylabel("Linear probe RMSE")
    ax.set_title("Decodability (RMSE) vs loss fit across all 99 runs\n"
                 "(does belief geometry just track how well the model fit?)")
    ax.legend(title="tree level")
    plt.tight_layout()
    out = os.path.join(OUTDIR, "arch_lossfit_rmse.png")
    plt.savefig(out)
    plt.close()
    print("wrote", out, os.path.getsize(out), "bytes")


# ---- 4. per-run sanity table (CSV, read by the typst appendix) -------------
def table_sanity():
    rows = []
    for r in RUNS:
        c, A = r["cfg"], r["A"]
        s = A["sanity"]
        rows.append({
            "grammar": c["grammar"], "seed": c["seed"],
            "n_layer": c["n_layer"], "n_embd": c["n_embd"],
            "n_head": c["n_head"], "steps": c["steps"],
            "test_ce": round(s["test_ce"], 3),
            "loss_gap_closed": round(s["loss_gap_closed"], 3),
            "root_r2": round(s["root_r2"], 3),
            "deepest_r2": round(s["deepest_r2"], 3),
            "root_rmse": round(s["root_rmse"], 3),
            "deepest_rmse": round(s["deepest_rmse"], 3),
        })
    df = (pd.DataFrame(rows)
          .sort_values(["n_layer", "n_embd", "n_head", "steps", "grammar", "seed"])
          .reset_index(drop=True))
    out = os.path.join(OUTDIR, "arch_sanity.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")
    for lvl in ("root_r2", "deepest_r2", "loss_gap_closed", "test_ce"):
        print(f"  {lvl}: mean={df[lvl].mean():.3f} sd={df[lvl].std():.3f} "
              f"min={df[lvl].min():.3f} max={df[lvl].max():.3f}")
    return df


if __name__ == "__main__":
    fig_marginal()
    fig_marginal_rmse()
    fig_depth_accum()
    fig_depth_accum_rmse()
    fig_lossfit()
    fig_lossfit_rmse()
    table_sanity()
