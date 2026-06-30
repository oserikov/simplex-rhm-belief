# Figures + table for the non-collapsing belief-geometry study (pass 5).
# Message: two grammar knobs -- ambiguity (rho, shared child-tuples across parents)
#   and skew (non-uniform rule choice). Ambiguity engineers a non-collapsing belief
#   geometry (k=8 root posterior stays uncertain); skew alone does NOT. The linear
#   probe still recovers the (now non-trivial) attractor.
# Data: globs results/noncollapse/*/{config.json,analysis.json,probe_data.npz}.
# Emits into paper-typst/figures/:
#   noncollapse_curve.png      - k=8 root entropy vs rho, one line per skew
#                                (skew>0,rho=0 = built-in control, stays ~0)
#   noncollapse_heatmap.png    - per-level probe R^2 over the skew x ambiguity grid
#                                (mean +/- std of the 3 draws), root and mid panels
#   noncollapse_attractor.png  - exact posterior + probe readout belief-PCA,
#                                uniform (none,none) vs high-ambiguity corner
#   noncollapse_sanity.csv     - per-run sanity table for the appendix
#   noncollapse_prereg.json    - scored pre-registered predictions
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

apply_style()

NC_DIRS = sorted(glob.glob(os.path.join(ROOT, "results/noncollapse/*")))
NC_DIRS = [d for d in NC_DIRS if os.path.isdir(d)]
assert NC_DIRS, "no results/noncollapse/* runs found"

SKEW_ORDER = ["none", "mid", "high"]
AMB_ORDER = [0.0, 0.3, 0.6]
CB = sns.color_palette("colorblind")
MID_KEYS = ["mid_L1a", "mid_L1b"]
LOW_KEYS = ["low_L2a", "low_L2b", "low_L2c", "low_L2d"]


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
        cfg = manifest["config"]
        nc = A.get("noncollapse", {})
        lv = A["latent_level_r2"]
        rows.append({
            "name": os.path.basename(d), "dir": d,
            "skew": cfg.get("skew", "none"),
            "ambiguity": float(cfg.get("ambiguity", 0.0)),
            "draw": int(cfg["grammar"]),
            "ent_by_pos": A["mean_posterior_entropy_by_position"],
            "root_r2": lv["root_L0"],
            "mid_r2": float(np.mean([lv[k] for k in MID_KEYS])),
            "low_r2": float(np.mean([lv[k] for k in LOW_KEYS])),
            "shuffled_r2": float(A["layer_r2_shuffled"][-1]),
            "k8_entropy": float(nc.get("root_entropy_full_context", np.nan)),
            "eff_dim": float(nc.get("reachable_eff_dim", np.nan)),
            "hull_area": float(nc.get("reachable_hull_area", np.nan)),
            "test_ce": A["sanity"]["test_ce"],
            "bayes_floor": A["sanity"]["bayes_floor"],
            "loss_gap_closed": A["sanity"]["loss_gap_closed"],
        })
    return pd.DataFrame(rows)


DF = load_runs(NC_DIRS)
assert len(DF) >= 2, f"need >=2 complete runs, got {len(DF)}"
print(f"loaded {len(DF)} complete noncollapse runs")


# ---- 1. non-collapse curve: k=8 entropy vs rho, one line per skew -----------
def fig_curve():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # left: k=8 endpoint entropy vs rho, one line per skew
    for i, sk in enumerate(SKEW_ORDER):
        sub = DF[DF["skew"] == sk]
        g = sub.groupby("ambiguity")
        m, sd = g.k8_entropy.mean(), g.k8_entropy.std().fillna(0)
        ax[0].errorbar(m.index.values, m.values, yerr=sd.values, marker="o", capsize=4,
                       color=CB[i], label=f"skew={sk}")
    ax[0].axhline(0, color="gray", lw=0.8, ls=":")
    ax[0].set_xlabel(r"ambiguity $\rho$"); ax[0].set_ylabel("mean $k{=}8$ root entropy (nats)")
    ax[0].set_title("Ambiguity engineers non-collapse\n($\\rho{=}0$ collapses for every skew)")
    ax[0].legend(title="rule-choice")
    # right: full entropy-vs-k trajectory at skew=none, one line per rho
    for j, am in enumerate(AMB_ORDER):
        sub = DF[(DF["skew"] == "none") & (np.isclose(DF["ambiguity"], am))]
        traj = np.stack(sub["ent_by_pos"].values)  # (draws, d)
        k = np.arange(1, traj.shape[1] + 1)
        ax[1].plot(k, traj.mean(0), marker="o", color=CB[j], label=f"ρ={am:g}")
        ax[1].fill_between(k, traj.mean(0) - traj.std(0), traj.mean(0) + traj.std(0),
                           color=CB[j], alpha=0.2)
    ax[1].axhline(0, color="gray", lw=0.8, ls=":")
    ax[1].set_xlabel("context position $k$"); ax[1].set_ylabel("mean root entropy (nats)")
    ax[1].set_title("Belief trajectory (skew=none)\ncollapses to 0 at $\\rho{=}0$, plateaus at $\\rho{>}0$")
    ax[1].legend(title="ambiguity")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "noncollapse_curve.png")
    fig.savefig(p); plt.close(fig); return p


# ---- 2. probe-robustness heatmap over the skew x ambiguity grid -------------
def _grid(metric):
    M = np.full((len(SKEW_ORDER), len(AMB_ORDER)), np.nan)
    S = np.full_like(M, np.nan)
    for i, sk in enumerate(SKEW_ORDER):
        for j, am in enumerate(AMB_ORDER):
            sub = DF[(DF["skew"] == sk) & (np.isclose(DF["ambiguity"], am))]
            if len(sub):
                M[i, j] = sub[metric].mean(); S[i, j] = sub[metric].std()
    return M, S


def fig_heatmap():
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for k, (metric, title) in enumerate([
        ("root_r2", "Root (L0) probe R²"),
        ("mid_r2", "Mid (L1) probe R²"),
        ("low_r2", "Low (L2) probe R²"),
    ]):
        M, S = _grid(metric)
        sh = DF.shuffled_r2.mean()
        annot = np.empty_like(M, dtype=object)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                annot[i, j] = f"{M[i, j]:.2f}\n±{S[i, j]:.2f}"
        sns.heatmap(M, annot=annot, fmt="", cmap="viridis", vmin=0, vmax=1,
                    xticklabels=[f"{a:g}" for a in AMB_ORDER],
                    yticklabels=SKEW_ORDER, ax=ax[k], cbar=k == 2,
                    annot_kws={"fontsize": 9})
        ax[k].set_title(f"{title}\n(shuffled baseline ≈ {sh:.02f})")
        ax[k].set_xlabel(r"ambiguity $\rho$"); ax[k].set_ylabel("skew" if k == 0 else "")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "noncollapse_heatmap.png")
    fig.savefig(p); plt.close(fig); return p


# ---- 3. attractor comparison: exact posterior + probe readout belief-PCA ----
def _corner(skew, amb):
    sub = DF[(DF["skew"] == skew) & (np.isclose(DF["ambiguity"], amb))].sort_values("draw")
    assert len(sub), f"no run for skew={skew} amb={amb}"
    return sub.iloc[0]["dir"]


def _readout(run_dir):
    """Belief-PCA of a run, in ITS OWN basis (root-class labels are not aligned
    across grammars, so each corner keeps its native basis). Returns the exact and
    probe-readout clouds, context position, and the 8 certainty-vertices + uniform
    prior projected into the same basis as absolute, label-agnostic references."""
    data = np.load(os.path.join(run_dir, "probe_data.npz"))
    hidden, beliefs = data["hidden"], data["beliefs"]  # (N,Lr,d,v_embd),(N,d,v)
    N, nL, d, ne = hidden.shape
    V = beliefs.shape[-1]
    Xf = hidden[:, nL - 1].reshape(N * d, ne)
    Y = beliefs.reshape(N * d, V)
    pos = np.tile(np.arange(1, d + 1), N)
    probe = LinearRegression().fit(Xf, Y)
    pca = PCA(n_components=2, random_state=0).fit(Y)
    verts = pca.transform(np.eye(V))                       # the 8 one-hot vertices
    prior = pca.transform(np.full((1, V), 1.0 / V))[0]     # uniform prior
    return pca.transform(Y), pca.transform(probe.predict(Xf)), pos, verts, prior, d


def fig_attractor():
    corners = [("none", 0.0, "Uniform unambiguous (ρ=0): collapses onto the vertices"),
               ("none", 0.6, "High ambiguity (ρ=0.6): stays in an interior cloud")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    for r, (sk, am, title) in enumerate(corners):
        exact, readout, pos, verts, prior, d = _readout(_corner(sk, am))
        # marker size: exponential, end-loaded growth so the LAST token dominates --
        # k=d is ~2x wider (~4x area) than k=1, growth accelerating toward the end.
        size = 10.0 * 4.0 ** ((pos - 1) / (d - 1))
        # alpha: dim the early/middle context (k<=6), bring out the last two tokens
        # (k=7 at 0.3, k=8 at 0.375).
        alpha = np.where(pos <= 6, 0.25, np.where(pos == 7, 0.3, 0.375))
        norm = plt.Normalize(1, d)
        order = np.argsort(pos)  # draw high-k last so the last token sits on top
        # shared limits WITHIN the row (exact & readout share this basis); framed on
        # the exact cloud + vertices so the corners are always in view.
        ref = np.vstack([exact, verts])
        (x0, y0), (x1, y1) = ref.min(0), ref.max(0)
        px, py = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
        for c, (coords, lab) in enumerate([(exact, "exact posterior"),
                                           (readout, "probe readout")]):
            a = ax[r, c]
            rgba = plt.cm.viridis(norm(pos[order]))
            rgba[:, 3] = alpha[order]  # per-point opacity
            a.scatter(coords[order, 0], coords[order, 1], c=rgba, s=size[order],
                      edgecolors="none")
            a.scatter(verts[:, 0], verts[:, 1], marker="o", s=150, facecolors="none",
                      edgecolors="black", linewidths=1.0, zorder=6,
                      label="certainty / simplex vertices")
            a.scatter([prior[0]], [prior[1]], marker="P", s=130, c="crimson",
                      edgecolors="white", linewidths=1.0, zorder=6, label="uniform prior")
            a.set_xlim(x0 - px, x1 + px); a.set_ylim(y0 - py, y1 + py)
            a.set_xlabel("belief PC1"); a.set_ylabel("belief PC2")
            a.set_title(f"{lab}\n{title}", fontsize=11)
            if r == 0 and c == 0:
                a.legend(loc="upper right", fontsize=8, framealpha=0.9)
            sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
            fig.colorbar(sm, ax=a, label="context position k")
    fig.tight_layout()
    p = os.path.join(OUTDIR, "noncollapse_attractor.png")
    fig.savefig(p); plt.close(fig); return p


# ---- 4. sanity table + scored pre-registration ------------------------------
def write_table():
    cols = ["skew", "ambiguity", "draw", "test_ce", "bayes_floor", "loss_gap_closed",
            "root_r2", "mid_r2", "low_r2", "shuffled_r2", "k8_entropy", "eff_dim",
            "hull_area"]
    out = DF[cols].sort_values(["skew", "ambiguity", "draw"])
    p = os.path.join(OUTDIR, "noncollapse_sanity.csv")
    out.to_csv(p, index=False, float_format="%.4f"); return p


def score_prereg():
    """Score the four pre-registered predictions against the 27-run grid."""
    def cell(sk, am, col):
        s = DF[(DF["skew"] == sk) & (np.isclose(DF["ambiguity"], am))][col]
        return float(s.mean())

    ent_none = {am: cell("none", am, "k8_entropy") for am in AMB_ORDER}
    # 1. rho>0 -> entropy>0 and monotone increasing (skew=none row)
    p1 = ent_none[0.0] < 1e-3 < ent_none[0.3] < ent_none[0.6]
    # 2. skew>0, rho=0 -> still collapses (entropy ~0)
    p2 = all(cell(sk, 0.0, "k8_entropy") < 1e-3 for sk in SKEW_ORDER)
    # 3. probe stays above shuffled baseline at every cell
    above = bool((DF.root_r2 > DF.shuffled_r2 + 0.02).all())
    # 3b. does root R2 degrade with rho? (prereg predicted yes -- report actual)
    r2_by_rho = {am: cell("none", am, "root_r2") for am in AMB_ORDER}
    degrades = r2_by_rho[0.6] < r2_by_rho[0.0]
    p3 = above
    out = {
        "pred1_ambiguity_noncollapse_monotone": bool(p1),
        "k8_entropy_skewnone_by_rho": ent_none,
        "pred2_skew_only_stays_collapsed": bool(p2),
        "k8_entropy_rho0_by_skew": {sk: cell(sk, 0.0, "k8_entropy") for sk in SKEW_ORDER},
        "pred3_probe_above_shuffled_everywhere": p3,
        "pred3_root_r2_degrades_with_rho": bool(degrades),
        "root_r2_skewnone_by_rho": r2_by_rho,
        "n_runs": int(len(DF)),
    }
    p = os.path.join(OUTDIR, "noncollapse_prereg.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("scored prereg:", json.dumps(out, indent=2))
    return p


if __name__ == "__main__":
    for fn in (fig_curve, fig_heatmap, fig_attractor, write_table, score_prereg):
        p = fn()
        sz = os.path.getsize(p)
        print(f"wrote {os.path.basename(p)} ({sz} bytes)")
        if p.endswith(".png"):
            assert sz > 10_000, f"{p} too small ({sz} bytes)"
