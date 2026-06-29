"""Probe + figures: residual stream -> belief simplex.

Consumes ``artifacts/probe_data.npz`` + ``artifacts/model.pt`` (from train.py).

1. Global linear (affine) probe: residual activations -> exact root-class
   posterior. Reports R^2 (held-out) vs a shuffled-label baseline, per layer
   and per context position.
2. Blooming geometry: PCA of the residual cloud showing the belief simplex
   image that expands with context position; layer-wise accumulation.
3. Causal steering (the chosen extra analysis): patch the final-layer residual
   along the probe direction for a root class and verify the next-token
   distribution shifts toward that class. Distinguishes a *used* representation
   from a merely *decodable* one.

Writes numbers to ``results/analysis.json`` and figures to ``results/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from rhm import Grammar
from train import build_model, get_device

ART = Path("artifacts")
RES = Path("results")
FIG = RES / "figures"


def load_grammar() -> Grammar:
    d = np.load(ART / "grammar.npz")
    L = int(d["L"])
    rules = [d[f"rules_{i}"] for i in range(L)]
    return Grammar(s=int(d["s"]), L=L, v=int(d["v"]), m=int(d["m"]), rules=rules)


def load_model(g: Grammar):
    ckpt = torch.load(ART / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(g, cfg["n_layer"], cfg["n_embd"], cfg["n_head"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def fit_probe(X_tr, Y_tr, X_te, Y_te):
    """Affine multi-output least squares; returns (probe, R2_test, R2_shuffled)."""
    reg = LinearRegression().fit(X_tr, Y_tr)
    r2 = reg.score(X_te, Y_te)
    rng = np.random.default_rng(0)
    Y_sh = Y_tr[rng.permutation(len(Y_tr))]
    reg_sh = LinearRegression().fit(X_tr, Y_sh)
    r2_sh = reg_sh.score(X_te, Y_te)
    return reg, r2, r2_sh


def main() -> None:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data = np.load(ART / "probe_data.npz")
    hidden = data["hidden"]          # (N, n_layers+1, d, n_embd)
    beliefs = data["beliefs"]        # (N, d, v)  exact root posterior
    N, n_layers, d, n_embd = hidden.shape
    v = beliefs.shape[-1]
    print(f"probe set N={N} layers={n_layers} d={d} n_embd={n_embd} v={v}")

    g = load_grammar()
    results: dict = {"meta": {"N": int(N), "n_layers": int(n_layers), "d": int(d),
                              "n_embd": int(n_embd), "v": int(v)}}

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_tr = int(N * 0.7)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    seqs = data["sequences"]

    # ---- 0. which latent does the residual encode? root vs deeper latents
    #         (spec: root is primary, mid-level optional). Final layer, pooled.
    def belief_tensor(node):
        Y = np.zeros((N, d, v))
        for i in range(N):
            for t in range(d):
                ell, pos = node
                Y[i, t] = (g.belief_root(seqs[i], t + 1) if ell == 0
                           else g.belief_node(seqs[i], t + 1, ell, pos))
        return Y

    level_nodes = {
        "root_L0": (0, 0),
        "mid_L1a": (1, 0),
        "mid_L1b": (1, 1),
        "low_L2a": (2, 0),
        "low_L2b": (2, 1),
        "low_L2c": (2, 2),
        "low_L2d": (2, 3),
    }
    level_r2 = {}
    Xf_rows_tr = np.concatenate([np.arange(i * d, i * d + d) for i in tr_idx])
    Xf_rows_te = np.concatenate([np.arange(i * d, i * d + d) for i in te_idx])
    Xfinal = hidden[:, n_layers - 1].reshape(N * d, n_embd)
    for name, node in level_nodes.items():
        Yn = belief_tensor(node).reshape(N * d, v)
        reg = LinearRegression().fit(Xfinal[Xf_rows_tr], Yn[Xf_rows_tr])
        level_r2[name] = float(reg.score(Xfinal[Xf_rows_te], Yn[Xf_rows_te]))
        print(f"latent {name}: R2={level_r2[name]:.4f}")
    results["latent_level_r2"] = level_r2

    # ---- 1. global probe per layer (pool all positions) ----------------
    layer_r2, layer_r2_sh = [], []
    probes = {}
    row_tr = np.concatenate([np.arange(i * d, i * d + d) for i in tr_idx])
    row_te = np.concatenate([np.arange(i * d, i * d + d) for i in te_idx])
    for ell in range(n_layers):
        X = hidden[:, ell].reshape(N * d, n_embd)
        Y = beliefs.reshape(N * d, v)
        reg, r2, r2_sh = fit_probe(X[row_tr], Y[row_tr], X[row_te], Y[row_te])
        probes[ell] = reg
        layer_r2.append(r2)
        layer_r2_sh.append(r2_sh)
        print(f"layer {ell}: R2={r2:.4f}  shuffled={r2_sh:.4f}")
    results["layer_r2"] = layer_r2
    results["layer_r2_shuffled"] = layer_r2_sh

    # ---- per-position R2 at the final layer ----------------------------
    final = n_layers - 1
    pos_r2 = []
    Xf = hidden[:, final]
    for t in range(d):
        Xt, Yt = Xf[:, t], beliefs[:, t]
        reg = LinearRegression().fit(Xt[tr_idx], Yt[tr_idx])
        pos_r2.append(float(reg.score(Xt[te_idx], Yt[te_idx])))
    results["position_r2_final_layer"] = pos_r2
    print("per-position R2 (final layer):", np.round(pos_r2, 3))

    # per (layer,position) R2 heatmap
    heat = np.zeros((n_layers, d))
    for ell in range(n_layers):
        for t in range(d):
            Xt, Yt = hidden[:, ell, t], beliefs[:, t]
            reg = LinearRegression().fit(Xt[tr_idx], Yt[tr_idx])
            heat[ell, t] = reg.score(Xt[te_idx], Yt[te_idx])
    results["heatmap_layer_position_r2"] = heat.tolist()

    # MAP classification accuracy (final-layer global probe, held-out rows)
    reg = probes[final]
    pred_te = reg.predict(hidden[:, final].reshape(N * d, n_embd)[row_te])
    acc = float(np.mean(pred_te.argmax(-1) == beliefs.reshape(N * d, v)[row_te].argmax(-1)))
    results["map_accuracy_final_layer"] = acc
    print(f"MAP root-class accuracy (final layer, held-out): {acc:.4f}")

    ent = np.array([[-np.sum(beliefs[i, t] * np.log(beliefs[i, t] + 1e-12))
                     for t in range(d)] for i in range(N)]).mean(0)
    results["mean_posterior_entropy_by_position"] = ent.tolist()

    # ============ FIGURES ============
    _fig_layer_position(layer_r2, layer_r2_sh, pos_r2, heat, FIG)
    _fig_blooming(hidden[:, final], beliefs, d, FIG)
    _fig_simplex_image(probes[final], hidden[:, final], beliefs, d, FIG)

    # ---- 3. causal steering --------------------------------------------
    steer = causal_steering(g, hidden)
    results["steering"] = steer
    _fig_steering(steer, FIG)

    (RES / "analysis.json").write_text(json.dumps(results, indent=2))
    print("wrote results/analysis.json and figures:", [p.name for p in FIG.iterdir()])


def _fig_layer_position(layer_r2, layer_r2_sh, pos_r2, heat, FIG):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(layer_r2))
    ax[0].plot(x, layer_r2, "o-", label="probe R²")
    ax[0].plot(x, layer_r2_sh, "s--", color="gray", label="shuffled baseline")
    ax[0].set_xlabel("layer (0=embed)"); ax[0].set_ylabel("R²")
    ax[0].set_title("Layer-wise accumulation"); ax[0].legend(); ax[0].set_xticks(x)
    p = np.arange(1, len(pos_r2) + 1)
    ax[1].plot(p, pos_r2, "o-", color="C2")
    ax[1].set_xlabel("context position k"); ax[1].set_ylabel("R²")
    ax[1].set_title("Decodability vs position")
    im = ax[2].imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax[2].set_xlabel("position"); ax[2].set_ylabel("layer")
    ax[2].set_title("R²(layer, position)"); fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(FIG / "probe_r2.png", dpi=150); plt.close(fig)


def _fig_blooming(Xf, beliefs, d, FIG):
    X = Xf.reshape(-1, Xf.shape[-1])
    pos = np.tile(np.arange(1, d + 1), Xf.shape[0])
    mapc = beliefs.reshape(-1, beliefs.shape[-1]).argmax(-1)
    coords = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = ax[0].scatter(coords[:, 0], coords[:, 1], c=pos, cmap="viridis", s=10, alpha=0.6)
    ax[0].set_title("Residual cloud (PCA) blooms with context position")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    fig.colorbar(sc, ax=ax[0], label="context position k")
    sc2 = ax[1].scatter(coords[:, 0], coords[:, 1], c=mapc, cmap="tab10", s=10, alpha=0.6)
    ax[1].set_title("Same cloud, colored by MAP root class")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    fig.colorbar(sc2, ax=ax[1], label="argmax root posterior")
    fig.tight_layout(); fig.savefig(FIG / "blooming_pca.png", dpi=150); plt.close(fig)


def _fig_simplex_image(probe, Xf, beliefs, d, FIG):
    X = Xf.reshape(-1, Xf.shape[-1])
    pred = probe.predict(X)
    true = beliefs.reshape(-1, beliefs.shape[-1])
    pos = np.tile(np.arange(1, d + 1), Xf.shape[0])
    pca = PCA(n_components=2).fit(true)
    pt, pp = pca.transform(true), pca.transform(pred)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    s0 = ax[0].scatter(pt[:, 0], pt[:, 1], c=pos, cmap="viridis", s=10, alpha=0.6)
    ax[0].set_title("Exact posterior simplex (PCA), colored by position")
    fig.colorbar(s0, ax=ax[0], label="position k")
    s1 = ax[1].scatter(pp[:, 0], pp[:, 1], c=pos, cmap="viridis", s=10, alpha=0.6)
    ax[1].set_title("Probe-predicted belief, same axes")
    fig.colorbar(s1, ax=ax[1], label="position k")
    for a in ax:
        a.set_xlabel("simplex PC1"); a.set_ylabel("simplex PC2")
    fig.tight_layout(); fig.savefig(FIG / "simplex_image.png", dpi=150); plt.close(fig)


def causal_steering(g, hidden, patch_layer=1, alphas=None):
    """Causal steering of the next-token's parent latent (mean-difference patch).

    The latent that *governs* the next token is its parent at level L-1. We patch
    the residual stream at the output of block ``patch_layer-1`` (so the remaining
    block can process the edited belief) by the class-mean difference
    ``alpha * (mu_target - mu_current)`` -- moving along the actual
    activation-manifold direction for that latent value. This is more faithful
    than adding a regression coefficient (wrong norm, washed out by the final
    layer-norm).

    Two metrics vs alpha:
      - log p(true next token): unchanged steering -> TRUE latent (already
        correct), collapses steering -> WRONG latent.
      - P(next token in the STEER TARGET latent's child set): rises as we install
        the target belief.
    A *used* representation shows both; a merely decodable one would not.
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
    device = get_device()
    model = load_model(g).to(device)
    data = np.load(ART / "probe_data.npz")
    sequences = data["sequences"]
    N, d = sequences.shape

    pv = np.zeros((N, d - 1), dtype=int)  # MAP parent latent of the next token
    for i in range(N):
        for t in range(d - 1):
            pv[i, t] = g.belief_node(sequences[i], k=t + 1, ell=g.L - 1, pos=(t + 1) // g.s).argmax()

    H = hidden[:, patch_layer]  # (N, d, n_embd)
    ne = H.shape[-1]
    mu = np.zeros((d - 1, g.v, ne))
    for t in range(d - 1):
        for c in range(g.v):
            sel = H[:, t][pv[:, t] == c]
            mu[t, c] = sel.mean(0) if len(sel) else H[:, t].mean(0)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    cur = torch.tensor(mu[np.arange(d - 1)[None].repeat(N, 0), pv], dtype=torch.float32, device=device)

    rng = np.random.default_rng(0)
    wrongv = np.array([[rng.choice([c for c in range(g.v) if c != pv[i, t]])
                        for t in range(d - 1)] for i in range(N)])

    seq_t = torch.tensor(sequences, dtype=torch.long, device=device)
    add = {"v": None}

    def hook(module, inp, out):
        if add["v"] is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs[:, : d - 1, :] = hs[:, : d - 1, :] + add["v"]
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

    handle = model.transformer.h[patch_layer - 1].register_forward_hook(hook)

    def run(alpha, tv):
        add["v"] = None
        with torch.no_grad():
            if alpha != 0.0:
                tgt = mu_t[torch.arange(d - 1)[None].repeat(N, 1), torch.tensor(tv, device=device)]
                add["v"] = alpha * (tgt - cur)
            logits = model(input_ids=seq_t).logits
        add["v"] = None
        logp = torch.log_softmax(logits[:, :-1], dim=-1).cpu().numpy()  # (N, d-1, v)
        true_lp = float(logp[np.arange(N)[:, None], np.arange(d - 1)[None], sequences[:, 1:]].mean())
        mass, cnt = 0.0, 0
        for i in range(N):
            for t in range(d - 1):
                child_pos = (t + 1) % g.s
                toks = list({int(g.rules[g.L - 1][tv[i, t], r, child_pos]) for r in range(g.m)})
                mass += float(np.exp(logp[i, t, toks]).sum())
                cnt += 1
        return true_lp, mass / cnt

    out = {"patch_layer": patch_layer, "alphas": list(alphas),
           "true_lp_steer_true": [], "true_lp_steer_wrong": [],
           "target_mass_steer_true": [], "target_mass_steer_wrong": []}
    for a in alphas:
        lt, mt = run(a, pv)
        lw, mw = run(a, wrongv)
        out["true_lp_steer_true"].append(lt)
        out["true_lp_steer_wrong"].append(lw)
        out["target_mass_steer_true"].append(mt)
        out["target_mass_steer_wrong"].append(mw)
    handle.remove()
    print("steer->true  log p(true):", np.round(out["true_lp_steer_true"], 3))
    print("steer->wrong log p(true):", np.round(out["true_lp_steer_wrong"], 3))
    print("steer->wrong P(target children):", np.round(out["target_mass_steer_wrong"], 3))
    return out


def _fig_steering(steer, FIG):
    a = steer["alphas"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(a, steer["true_lp_steer_true"], "o-", color="C2", label="steer → TRUE latent")
    ax[0].plot(a, steer["true_lp_steer_wrong"], "s--", color="C3", label="steer → WRONG latent")
    ax[0].set_xlabel("patch strength α"); ax[0].set_ylabel("mean log p(true next token)")
    ax[0].set_title("Corrupting the belief breaks prediction"); ax[0].legend()
    ax[1].plot(a, steer["target_mass_steer_wrong"], "s-", color="C3",
               label="steer → wrong (target)")
    ax[1].plot(a, steer["target_mass_steer_true"], "o-", color="C2",
               label="steer → true (control)")
    ax[1].set_xlabel("patch strength α"); ax[1].set_ylabel("P(next token ∈ target latent's children)")
    ax[1].set_title("Steering installs the target belief"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "steering.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
