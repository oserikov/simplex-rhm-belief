"""Exp 1 follow-up - Forest-RHM interleave layout, model-width sweep.

Tests whether the interleave layout's RMSE plateau (0.12-0.24 at n_embd=128,
see results/exp_forest/sanity.json) is a width-dependent superposition
bottleneck. Naive capacity requirement is k(v-1)=28 dims, so n_embd=128 gives
only ~4.5x slack -- far below the ~30-50x Shai et al. use for their
near-zero results, while concat (same capacity requirement) already hits
near-zero at n_embd=128. Sweeps n_embd in {128, 256, 512, 1024} at fixed
k=4, s=2, v=8, m=2, interleave layout only.

Pre-registered prediction: interleave per-tree RMSE drops toward concat's
near-zero as width grows. If it still plateaus at 512-1024, the bottleneck
is not width -- report faithfully either way.

Emits results/exp_forest/width_sweep.json and
results/figures/forest_width_sweep_rmse.png. Leaves results/exp_forest/
sanity.json untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from analyze import rmse
from rhm_variants import ForestGrammar, brute_force_forest_belief
from train import train

RES = Path("results/exp_forest")
ART = Path("artifacts/exp_forest_width")
FIG = Path("results/figures")

K, S, V, M = 4, 2, 8, 2
LAYOUT = "interleave"
WIDTHS = [128, 256, 512, 1024]
NEAR_ONE_HOT = 0.99
CONCAT_REFERENCE_RMSE = 0.03  # committed concat run's near-zero pass threshold


def quick_checks(fg: ForestGrammar) -> None:
    rng = np.random.default_rng(1)
    x, _ = fg.sample(rng)
    assert len(x) == fg.k * fg.s, "sample length != k*s"
    for k_obs in range(0, fg.d + 1, 2):
        b = fg.belief_forest(x, k_obs)
        bf = brute_force_forest_belief(fg, x, k_obs)
        err = np.abs(b - bf).max()
        assert err < 1e-6, f"belief_forest vs brute force mismatch {err}"
    per = fg.split_leaves(x)
    b0 = fg.trees[0].belief_root(per[0], fg.observed_count(0, fg.d))
    per2 = [p.copy() for p in per]
    per2[1] = (per2[1] + 1) % fg.v
    b0_after = fg.trees[0].belief_root(per[0], fg.observed_count(0, fg.d))
    assert np.abs(b0 - b0_after).max() < 1e-9
    print("[exp_forest_width] quick-failure checkpoints: PASS")


# (lr, steps, early_stop) per width. 128/256 keep the original run's fixed-step,
# no-early-stop config so results reproduce results/exp_forest/sanity.json exactly.
# 512/1024 use a lower lr + early-stopping (restore best held-out-loss checkpoint):
# the original run's fixed lr=3e-3/no-early-stop combo let wider models overshoot
# late in training (final loss worse than an earlier checkpoint), which is what
# actually caused their loss_gap_closed collapse -- not an insufficient step count.
# Chosen by a short lr/step probe (see WORKLOG-equivalent commit message) confirming
# each config reaches loss_gap_closed >= ~0.98 well within the original step budget.
WIDTH_CONFIG = {
    128: dict(lr=3e-3, steps=6000, early_stop=False),
    256: dict(lr=3e-3, steps=6000, early_stop=False),
    512: dict(lr=5e-4, steps=6000, early_stop=True),
    1024: dict(lr=5e-4, steps=8000, early_stop=True),
}


def make_args(seed: int, n_embd: int) -> SimpleNamespace:
    cfg = WIDTH_CONFIG[n_embd]
    return SimpleNamespace(
        s=S, L=1, v=V, m=M, grammar_seed=seed, seed=seed,
        ambiguity=0.0, skew="none",
        n_layer=2, n_embd=n_embd, n_head=4, lr=cfg["lr"], steps=cfg["steps"],
        batch_size=128, test_frac=0.2, n_probe=3000, log_every=500,
        early_stop=cfg["early_stop"],
    )


def probe_row_space(coef: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal basis (n_embd, k) for the top-k row-space directions of a
    (v, n_embd) regression coefficient matrix (each row is a probe direction
    in R^n_embd; the belief simplex constraint caps the informative rank at
    v-1, so k=v-1 captures the full row space in the typical case)."""
    u, _, _ = np.linalg.svd(coef.T, full_matrices=False)
    return u[:, :k]


def run_width(n_embd: int, seed: int = 0) -> dict:
    print(f"\n=== Exp 1 width sweep: layout={LAYOUT} n_embd={n_embd} ===")
    fg = ForestGrammar.random(k=K, layout=LAYOUT, s=S, v=V, m=M, seed=seed)
    if n_embd == WIDTHS[0]:
        quick_checks(fg)

    cap_needed = K * (V - 1)
    slack = n_embd / cap_needed
    print(f"[exp_forest_width] capacity check: k(v-1)={cap_needed} n_embd={n_embd} "
          f"slack={slack:.1f}x")

    out_dir = ART / f"n{n_embd}"
    args = make_args(seed, n_embd)
    summary = train(args, out_dir=out_dir, grammar=fg)

    data = np.load(out_dir / "probe_data.npz")
    hidden = data["hidden"]
    sequences = data["sequences"]
    N, n_layers, d, ne = hidden.shape
    final = n_layers - 1

    beliefs_per_tree = np.zeros((N, K, V))
    onehot_checks = []
    for i in range(N):
        beliefs_per_tree[i] = fg.belief_forest(sequences[i], fg.d)
        b_early = fg.belief_forest(sequences[i], 1)
        onehot_checks.append(b_early.max(axis=-1) > NEAR_ONE_HOT)
    onehot_checks = np.array(onehot_checks)
    one_hot_fraction_after_1_leaf = float(onehot_checks.mean())

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = int(N * 0.7)
    tr, te = perm[:n_tr], perm[n_tr:]

    own_position = []
    for j in range(K):
        k_obs = next(k for k in range(1, fg.d + 1)
                     if fg.observed_count(j, k) == fg.trees[j].d)
        own_position.append(k_obs - 1)

    per_tree_rmse, per_tree_rmse_sh, own_position_trained = [], [], []
    coefs = {}
    for j in range(K):
        pos = own_position[j]
        Xj = hidden[:, final, pos, :]
        Y = beliefs_per_tree[:, j, :]
        reg = LinearRegression().fit(Xj[tr], Y[tr])
        coefs[j] = reg.coef_  # (v, n_embd)
        pred = reg.predict(Xj[te])
        r = rmse(pred, Y[te])
        rng2 = np.random.default_rng(0)
        Y_sh = Y[tr][rng2.permutation(len(tr))]
        reg_sh = LinearRegression().fit(Xj[tr], Y_sh)
        r_sh = rmse(reg_sh.predict(Xj[te]), Y[te])
        per_tree_rmse.append(r)
        per_tree_rmse_sh.append(r_sh)
        trained = pos < d - 1
        own_position_trained.append(trained)
        print(f"[exp_forest_width] n_embd={n_embd} tree {j} root RMSE={r:.4f} "
              f"(shuffled baseline={r_sh:.4f}) @ position={pos} trained={trained}")

    # cross-tree leakage (interleave: leak_pos = K-1)
    leak_pos = K - 1
    assert fg.observed_count(K - 1, leak_pos) == 0, "leakage probe position leaks target tree"
    leak_layer_idx = leak_pos - 1
    Xleak = hidden[:, final, leak_layer_idx, :]
    Yleak = beliefs_per_tree[:, K - 1, :]
    reg_leak = LinearRegression().fit(Xleak[tr], Yleak[tr])
    leak_rmse = rmse(reg_leak.predict(Xleak[te]), Yleak[te])
    rng3 = np.random.default_rng(0)
    Yleak_sh = Yleak[tr][rng3.permutation(len(tr))]
    reg_leak_sh = LinearRegression().fit(Xleak[tr], Yleak_sh)
    leak_rmse_sh = rmse(reg_leak_sh.predict(Xleak[te]), Yleak[te])
    print(f"[exp_forest_width] n_embd={n_embd} cross-tree leakage RMSE={leak_rmse:.4f} "
          f"(shuffled baseline={leak_rmse_sh:.4f})")

    # probe-subspace orthogonality diagnostic
    k_dim = V - 1
    overlaps = {}
    cos_all = []
    for i in range(K):
        for j in range(i + 1, K):
            Ui = probe_row_space(coefs[i], k_dim)
            Uj = probe_row_space(coefs[j], k_dim)
            sv = np.linalg.svd(Ui.T @ Uj, compute_uv=False)
            mean_abs_cos = float(np.mean(np.abs(sv)))
            overlaps[f"{i}-{j}"] = mean_abs_cos
            cos_all.append(mean_abs_cos)
    mean_overlap = float(np.mean(cos_all))
    print(f"[exp_forest_width] n_embd={n_embd} probe-subspace mean |cos| overlap="
          f"{mean_overlap:.4f} (0=orthogonal, 1=identical)")

    result = {
        "n_embd": n_embd,
        "capacity_needed": cap_needed,
        "slack": slack,
        "lr": WIDTH_CONFIG[n_embd]["lr"],
        "steps": WIDTH_CONFIG[n_embd]["steps"],
        "early_stop": WIDTH_CONFIG[n_embd]["early_stop"],
        "train_summary": {k: summary[k] for k in
                           ("final_test_loss", "uniform_baseline",
                            "bayes_optimal_mean", "loss_gap_closed")},
        "per_tree_root_rmse": per_tree_rmse,
        "per_tree_root_rmse_shuffled": per_tree_rmse_sh,
        "own_position": own_position,
        "own_position_trained": own_position_trained,
        "pass_threshold_0.03_trained_only": [r <= 0.03 for r, t in
                                              zip(per_tree_rmse, own_position_trained, strict=True) if t],
        "one_hot_fraction_after_1_leaf": one_hot_fraction_after_1_leaf,
        "cross_tree_leakage_rmse": leak_rmse,
        "cross_tree_leakage_rmse_shuffled": leak_rmse_sh,
        "cross_tree_leakage_at_baseline": bool(abs(leak_rmse - leak_rmse_sh) < 0.02),
        "probe_subspace_overlap_pairwise": overlaps,
        "probe_subspace_overlap_mean": mean_overlap,
    }
    return result


def make_figure(rows: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    widths = [r["n_embd"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    n_trees = len(rows[0]["per_tree_root_rmse"])
    for j in range(n_trees):
        vals = [r["per_tree_root_rmse"][j] for r in rows]
        trained = [r["own_position_trained"][j] for r in rows]
        marker = "o-" if all(trained) else "o--"
        ax[0].plot(widths, vals, marker, label=f"tree {j}"
                   + ("" if all(trained) else " (last tree, untrained pos)"))
    ax[0].axhspan(0, CONCAT_REFERENCE_RMSE, color="green", alpha=0.15,
                  label="concat near-zero band (<=0.03)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(widths)
    ax[0].set_xticklabels(widths)
    ax[0].set_xlabel("n_embd")
    ax[0].set_ylabel("per-tree root RMSE")
    ax[0].set_title("Interleave: RMSE vs width")
    ax[0].legend(fontsize=7)

    overlap = [r["probe_subspace_overlap_mean"] for r in rows]
    gap = [r["train_summary"]["loss_gap_closed"] for r in rows]
    ax2 = ax[1].twinx()
    p1, = ax[1].plot(widths, overlap, "o-", color="C0", label="mean probe-subspace |cos| overlap")
    p2, = ax2.plot(widths, gap, "s--", color="C3", label="loss_gap_closed")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(widths)
    ax[1].set_xticklabels(widths)
    ax[1].set_xlabel("n_embd")
    ax[1].set_ylabel("mean |cos| between tree-pair probe subspaces", color="C0")
    ax2.set_ylabel("loss_gap_closed", color="C3")
    ax[1].set_title("Superposition diagnostic + generalization check")
    ax[1].legend(handles=[p1, p2], fontsize=7, loc="center right")

    fig.tight_layout()
    fig.savefig(FIG / "forest_width_sweep_rmse.png", dpi=150)
    plt.close(fig)
    print(f"[exp_forest_width] wrote {FIG}/forest_width_sweep_rmse.png")


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    rows = [run_width(n_embd) for n_embd in WIDTHS]
    (RES / "width_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[exp_forest_width] wrote {RES}/width_sweep.json")
    make_figure(rows)
    print("\n[exp_forest_width] summary:")
    for r in rows:
        print(f"  n_embd={r['n_embd']:5d} slack={r['slack']:.1f}x "
              f"per_tree_rmse={np.round(r['per_tree_root_rmse'], 4)} "
              f"loss_gap_closed={r['train_summary']['loss_gap_closed']:.4f} "
              f"mean_overlap={r['probe_subspace_overlap_mean']:.4f}")


if __name__ == "__main__":
    main()
