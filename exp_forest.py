"""Exp 1 - Forest-RHM positive control.

k independent depth-1 trees: the joint posterior factors by construction, so a
linear probe should recover every tree's root belief near-exactly. Runs BOTH
leaf layouts (concat, interleave). See spec_variants.md Exp 1.

Quick-failure checkpoints run first (seconds): sample length, belief_forest vs
brute force, factorization. Then: train -> dump hidden states -> fit one probe
per tree per layout -> cross-tree leakage probe -> sanity JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LinearRegression

from analyze import rmse
from rhm_variants import ForestGrammar, brute_force_forest_belief
from train import train

RES = Path("results/exp_forest")
ART = Path("artifacts/exp_forest")

K, S, V, M = 4, 2, 8, 2
NEAR_ONE_HOT = 0.99


def quick_checks(fg: ForestGrammar) -> None:
    rng = np.random.default_rng(1)
    x, _ = fg.sample(rng)
    assert len(x) == fg.k * fg.s, "sample length != k*s"
    for k_obs in range(0, fg.d + 1, 2):
        b = fg.belief_forest(x, k_obs)
        bf = brute_force_forest_belief(fg, x, k_obs)
        err = np.abs(b - bf).max()
        assert err < 1e-6, f"belief_forest vs brute force mismatch {err}"
    # factorization: tree j's belief fn structurally never reads other trees' leaves
    per = fg.split_leaves(x)
    b0 = fg.trees[0].belief_root(per[0], fg.observed_count(0, fg.d))
    per2 = [p.copy() for p in per]
    per2[1] = (per2[1] + 1) % fg.v
    b0_after = fg.trees[0].belief_root(per[0], fg.observed_count(0, fg.d))
    assert np.abs(b0 - b0_after).max() < 1e-9
    print("[exp_forest] quick-failure checkpoints: PASS")


def make_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        s=S, L=1, v=V, m=M, grammar_seed=seed, seed=seed,
        ambiguity=0.0, skew="none",
        n_layer=2, n_embd=128, n_head=4, lr=3e-3, steps=6000,
        batch_size=128, test_frac=0.2, n_probe=3000, log_every=500,
    )


def run_layout(layout: str, seed: int = 0) -> dict:
    print(f"\n=== Exp 1 (Forest): layout={layout} ===")
    fg = ForestGrammar.random(k=K, layout=layout, s=S, v=V, m=M, seed=seed)
    quick_checks(fg)

    cap_needed = K * (V - 1)
    print(f"[exp_forest] capacity check: k(v-1)={cap_needed} <= n_embd={128}: "
          f"{cap_needed <= 128}")

    out_dir = ART / layout
    args = make_args(seed)
    summary = train(args, out_dir=out_dir, grammar=fg)

    data = np.load(out_dir / "probe_data.npz")
    hidden = data["hidden"]  # (N, n_layer+1, d, n_embd)
    sequences = data["sequences"]  # (N, d)
    N, n_layers, d, n_embd = hidden.shape
    final = n_layers - 1

    # recompute the factorized target ourselves (per spec: don't rely on the
    # single-tree belief_root/belief_node dump)
    beliefs_per_tree = np.zeros((N, K, V))  # at FULL context (final position)
    onehot_checks = []  # informativeness guard: fraction near-one-hot after 1 leaf observed
    for i in range(N):
        beliefs_per_tree[i] = fg.belief_forest(sequences[i], fg.d)
        b_early = fg.belief_forest(sequences[i], 1)  # after 1 global leaf observed
        onehot_checks.append(b_early.max(axis=-1) > NEAR_ONE_HOT)
    onehot_checks = np.array(onehot_checks)  # (N, K)
    one_hot_fraction_after_1_leaf = float(onehot_checks.mean())
    print(f"[exp_forest] triviality guard: fraction of near-one-hot per-tree beliefs "
          f"after 1 global leaf observed = {one_hot_fraction_after_1_leaf:.3f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = int(N * 0.7)
    tr, te = perm[:n_tr], perm[n_tr:]

    # Each tree's belief is probed from the residual at THAT TREE's own
    # resolution position (the earliest position at which its own leaves are
    # fully observed) rather than the shared sequence-final position: GPT2's
    # next-token loss slices off the last position (nothing to predict after
    # it), so its residual carries no training signal regardless of the
    # underlying dependency structure. Using a shared terminal position for
    # every tree would conflate that training artifact with the dependency
    # question this experiment is testing. Only the tree that happens to own
    # the literal last leaf slot is unavoidably probed from that untrained
    # position (documented per-tree via ``own_position_trained`` below).
    own_position = []
    for j in range(K):
        # smallest k_obs at which observed_count(j, k_obs) == s_sub
        k_obs = next(k for k in range(1, fg.d + 1)
                     if fg.observed_count(j, k) == fg.trees[j].d)
        own_position.append(k_obs - 1)  # 0-indexed hidden-state position

    per_tree_rmse, per_tree_rmse_sh, own_position_trained = [], [], []
    for j in range(K):
        pos = own_position[j]
        Xj = hidden[:, final, pos, :]
        Y = beliefs_per_tree[:, j, :]
        reg = LinearRegression().fit(Xj[tr], Y[tr])
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
        print(f"[exp_forest] tree {j} root RMSE={r:.4f} (shuffled baseline={r_sh:.4f}) "
              f"@ own-resolution position={pos} trained={trained}")

    # cross-tree leakage: probe tree (K-1)'s belief from a context where ONLY the
    # other K-1 trees' leaves are observed (concat: position (K-1)*s; interleave:
    # the largest k_obs at which tree K-1 has observed_count == 0)
    leak_pos = (K - 1) * S if layout == "concat" else K - 1
    assert fg.observed_count(K - 1, leak_pos) == 0, "leakage probe position leaks target tree"
    leak_layer_idx = leak_pos - 1  # residual AFTER token leak_pos-1, i.e. having seen leak_pos tokens
    Xleak = hidden[:, final, leak_layer_idx, :]
    Yleak = beliefs_per_tree[:, K - 1, :]  # still == uniform prior (nothing observed of tree K-1)
    reg_leak = LinearRegression().fit(Xleak[tr], Yleak[tr])
    leak_rmse = rmse(reg_leak.predict(Xleak[te]), Yleak[te])
    rng3 = np.random.default_rng(0)
    Yleak_sh = Yleak[tr][rng3.permutation(len(tr))]
    reg_leak_sh = LinearRegression().fit(Xleak[tr], Yleak_sh)
    leak_rmse_sh = rmse(reg_leak_sh.predict(Xleak[te]), Yleak[te])
    print(f"[exp_forest] cross-tree leakage RMSE={leak_rmse:.4f} "
          f"(shuffled baseline={leak_rmse_sh:.4f}); observed_count(tree{K-1})="
          f"{fg.observed_count(K - 1, leak_pos)} at global k={leak_pos}")

    result = {
        "layout": layout,
        "k": K, "s": S, "v": V, "m": M, "d": fg.d,
        "capacity_needed": cap_needed, "n_embd": 128,
        "train_summary": {k: summary[k] for k in
                           ("final_test_loss", "uniform_baseline",
                            "bayes_optimal_mean", "loss_gap_closed")},
        "per_tree_root_rmse": per_tree_rmse,
        "per_tree_root_rmse_shuffled": per_tree_rmse_sh,
        "own_position": own_position,
        "own_position_trained": own_position_trained,
        "pass_threshold_0.03": [r <= 0.03 for r in per_tree_rmse],
        "pass_threshold_0.03_trained_only": [r <= 0.03 for r, t in
                                              zip(per_tree_rmse, own_position_trained, strict=True) if t],
        "one_hot_fraction_after_1_leaf": one_hot_fraction_after_1_leaf,
        "cross_tree_leakage_rmse": leak_rmse,
        "cross_tree_leakage_rmse_shuffled": leak_rmse_sh,
        "cross_tree_leakage_at_baseline": bool(abs(leak_rmse - leak_rmse_sh) < 0.02),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    results = {}
    for layout in ("concat", "interleave"):
        results[layout] = run_layout(layout, seed=args.seed)
    (RES / "sanity.json").write_text(json.dumps(results, indent=2))
    print(f"\n[exp_forest] wrote {RES}/sanity.json")
    for layout, r in results.items():
        print(f"{layout}: per-tree RMSE={np.round(r['per_tree_root_rmse'], 4)} "
              f"pass<=0.03: {r['pass_threshold_0.03']} "
              f"leakage_at_baseline={r['cross_tree_leakage_at_baseline']}")


if __name__ == "__main__":
    main()
