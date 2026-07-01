"""Exp 2 - epsilon-decoupled RHM: RMSE-vs-dependence sweep.

Sweeps eps in {0, 0.25, 0.5, 0.75, 1.0}. eps dials ONLY the top-level
dependence between the two subtree-roots (nodes (1,0), (1,1)); everything
below is byte-identical across the sweep. One deep node (2,0), sitting inside
an untouched subtree, is the built-in negative control (expected flat).
Root belief is vacuous at eps>0 and excluded from success metrics (spec's root
caveat). See spec_variants.md Exp 2.

Emits results/exp_epsilon/sanity.csv and results/figures/epsilon_sweep_rmse.png.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from analyze import rmse
from rhm import Grammar
from rhm_variants import EpsilonGrammar, brute_force_epsilon_node_belief
from train import train

RES = Path("results/exp_epsilon")
ART = Path("artifacts/exp_epsilon")
FIG = Path("results/figures")

S, L, V, M = 2, 3, 8, 2
EPS_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]


def quick_checks(base: Grammar) -> None:
    rng = np.random.default_rng(2)
    x, _ = base.sample(rng)

    eg0 = EpsilonGrammar(base=base, eps=0.0)
    for k in [1, 3, 5, 8]:
        for pos in (0, 1):
            got = eg0.belief_node(x, k, 1, pos)
            want = base.belief_node(x, k, 1, pos)
            assert np.allclose(got, want, atol=1e-9), f"eps=0 mismatch k={k} pos={pos}"

    eg1 = EpsilonGrammar(base=base, eps=1.0)
    for k in [0, 1, 2, 4, 8]:
        mu = base._upward(x, k)
        joint = eg1.joint_prior_mix * np.outer(mu[(1, 0)], mu[(1, 1)])
        joint = joint / joint.sum()
        post0 = eg1.belief_node(x, k, 1, 0)
        post1 = eg1.belief_node(x, k, 1, 1)
        assert np.abs(joint - np.outer(post0, post1)).max() < 1e-9, f"eps=1 factorization fails k={k}"

    eg5 = EpsilonGrammar(base=base, eps=0.5)
    for k in [0, 1, 2]:
        for ell, pos in [(1, 0), (1, 1), (2, 0)]:
            got = eg5.belief_node(x, k, ell, pos)
            want = brute_force_epsilon_node_belief(eg5, x, k, ell, pos)
            assert np.abs(got - want).max() < 1e-6, f"BP vs brute force mismatch k={k} ell={ell} pos={pos}"

    print("[exp_epsilon] quick-failure checkpoints: PASS")


def make_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        s=S, L=L, v=V, m=M, grammar_seed=seed, seed=seed,
        ambiguity=0.0, skew="none",
        n_layer=2, n_embd=128, n_head=4, lr=3e-3, steps=6000,
        batch_size=128, test_frac=0.2, n_probe=3000, log_every=1000,
    )


def fit_target_rmse(hidden, sequences, eg, node, final_layer):
    """Pooled (all N, all positions) probe RMSE for one belief target, final layer."""
    N, d = sequences.shape
    ell, pos = node
    Y = np.zeros((N, d, eg.v))
    for i in range(N):
        for t in range(d):
            Y[i, t] = eg.belief_node(sequences[i], t + 1, ell, pos)
    X = hidden[:, final_layer].reshape(N * d, -1)
    Yf = Y.reshape(N * d, eg.v)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N * d)
    n_tr = int(len(perm) * 0.7)
    tr, te = perm[:n_tr], perm[n_tr:]
    reg = LinearRegression().fit(X[tr], Yf[tr])
    return rmse(reg.predict(X[te]), Yf[te])


def cross_leak_diagnostic(eg: EpsilonGrammar, hidden, sequences, final_layer) -> tuple[float, float]:
    """Isolates the cross-subtree-dependence effect the pooled metric dilutes:
    at position k=4 (subtree0's 4 leaves fully observed, subtree1 untouched),
    probe node(1,1)'s belief from ONLY cross-subtree evidence. At eps=0 this
    target is a genuine (non-constant) function of subtree0's resolved root;
    at eps=1 it collapses to the eps-independent marginal (constant across
    examples) since no cross-information exists by construction. Returns
    (probe RMSE, mean target entropy) -- entropy captures the information
    content directly, decoupled from whether the probe decodes it."""
    N = sequences.shape[0]
    pos = 3  # k_obs = 4
    Y = np.array([eg.belief_node(sequences[i], pos + 1, 1, 1) for i in range(N)])
    X = hidden[:, final_layer, pos, :]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = int(N * 0.7)
    tr, te = perm[:n_tr], perm[n_tr:]
    reg = LinearRegression().fit(X[tr], Y[tr])
    r = rmse(reg.predict(X[te]), Y[te])
    ent = float(np.mean([-np.sum(y * np.log(y + 1e-12)) for y in Y]))
    return r, ent


def run_sweep(seed: int = 0) -> list[dict]:
    base = Grammar.random(s=S, L=L, v=V, m=M, seed=seed)
    quick_checks(base)

    rows = []
    for eps in EPS_SWEEP:
        print(f"\n=== Exp 2 (Epsilon): eps={eps} ===")
        eg = EpsilonGrammar(base=base, eps=eps)
        out_dir = ART / f"eps_{eps}"
        args = make_args(seed)
        summary = train(args, out_dir=out_dir, grammar=eg)

        data = np.load(out_dir / "probe_data.npz")
        hidden = data["hidden"]
        sequences = data["sequences"]
        n_layers = hidden.shape[1]
        final = n_layers - 1

        cross_cut_a = fit_target_rmse(hidden, sequences, eg, (1, 0), final)
        cross_cut_b = fit_target_rmse(hidden, sequences, eg, (1, 1), final)
        cross_cut_rmse = float(np.mean([cross_cut_a, cross_cut_b]))
        within_subtree_rmse = fit_target_rmse(hidden, sequences, eg, (2, 0), final)
        leak_rmse, leak_entropy = cross_leak_diagnostic(eg, hidden, sequences, final)

        print(f"[exp_epsilon] eps={eps}: cross-cut RMSE={cross_cut_rmse:.4f} "
              f"(node10={cross_cut_a:.4f}, node11={cross_cut_b:.4f}) "
              f"within-subtree RMSE={within_subtree_rmse:.4f} "
              f"cross-leak-diagnostic RMSE={leak_rmse:.4f} target-entropy={leak_entropy:.4f}")
        rows.append({
            "eps": eps,
            "cross_cut_rmse": cross_cut_rmse,
            "cross_cut_node10_rmse": cross_cut_a,
            "cross_cut_node11_rmse": cross_cut_b,
            "within_subtree_rmse": within_subtree_rmse,
            "cross_leak_diagnostic_rmse": leak_rmse,
            "cross_leak_diagnostic_target_entropy": leak_entropy,
            "final_test_loss": summary["final_test_loss"],
            "uniform_baseline": summary["uniform_baseline"],
            "bayes_optimal_mean": summary["bayes_optimal_mean"],
        })
    return rows


def write_sanity_csv(rows: list[dict]) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[exp_epsilon] wrote {RES}/sanity.csv")


def make_figure(rows: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    eps = [r["eps"] for r in rows]
    cross = [r["cross_cut_rmse"] for r in rows]
    within = [r["within_subtree_rmse"] for r in rows]
    leak_rmse = [r["cross_leak_diagnostic_rmse"] for r in rows]
    leak_ent = [r["cross_leak_diagnostic_target_entropy"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(eps, cross, "o-", color="C0", label="subtree-root (cross-cut, pooled)")
    ax[0].plot(eps, within, "s--", color="C1", label="deep node (2,0) (within-subtree, control)")
    ax[0].set_xlabel(r"$\epsilon$ (decoupling rate)")
    ax[0].set_ylabel("probe RMSE")
    ax[0].set_title("RMSE tracks dependence, not difficulty")
    ax[0].legend(fontsize=8)

    ax2 = ax[1].twinx()
    p1, = ax[1].plot(eps, leak_rmse, "o-", color="C2", label="cross-leak probe RMSE")
    p2, = ax2.plot(eps, leak_ent, "^--", color="C3", label="cross-leak target entropy (true)")
    ax[1].set_xlabel(r"$\epsilon$ (decoupling rate)")
    ax[1].set_ylabel("probe RMSE", color="C2")
    ax2.set_ylabel("true target entropy (nats)", color="C3")
    ax[1].set_title("Isolated cross-subtree leak (k=4, node (1,1))")
    ax[1].legend(handles=[p1, p2], fontsize=8, loc="center left")

    fig.tight_layout()
    fig.savefig(FIG / "epsilon_sweep_rmse.png", dpi=150)
    plt.close(fig)
    print(f"[exp_epsilon] wrote {FIG}/epsilon_sweep_rmse.png")


def main() -> None:
    rows = run_sweep()
    write_sanity_csv(rows)
    make_figure(rows)
    print("\n[exp_epsilon] summary:")
    for r in rows:
        print(f"  eps={r['eps']}: cross-cut={r['cross_cut_rmse']:.4f} "
              f"within-subtree={r['within_subtree_rmse']:.4f}")


if __name__ == "__main__":
    main()
