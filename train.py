"""Train a tiny decoder-only transformer on RHM samples (next-token CE).

Default: 2 layers, d_model 128, 4 heads, context = d = s ** L = 8. Dropout off.
No tokenizer -- RHM leaf symbols are fed straight as ``input_ids``.

Saves to ``artifacts/``:
- ``model.pt``         : model weights + config
- ``grammar.npz``      : grammar rules + params (so analyze.py reconstructs it)
- ``probe_data.npz``   : per-(example, layer, position) residual activations
                         paired with the exact root-class belief state.

Device: PyTorch MPS with CPU fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from rhm import Grammar

ART = Path("artifacts")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(g: Grammar, n_layer: int, n_embd: int, n_head: int) -> GPT2LMHeadModel:
    cfg = GPT2Config(
        vocab_size=g.v,
        n_positions=g.d,
        n_ctx=g.d,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=None,  # silence out-of-vocab default token warnings
        eos_token_id=None,
    )
    return GPT2LMHeadModel(cfg)


def conditional_entropy_floor(g: Grammar) -> np.ndarray:
    """Exact average next-token conditional entropy H(x_{k+1}|x_1..x_k) per position.

    This is the Bayes-optimal next-token loss the model can reach. Uses exact
    enumeration of all trees.
    """
    leaves, _ = g.enumerate_all()
    n = len(leaves)
    floor = np.zeros(g.d - 1)
    # group by prefix to get next-token distributions
    for k in range(1, g.d):
        # P(next | prefix) from empirical (= exact, equiprobable) over trees
        from collections import defaultdict

        buckets: dict[tuple, list[int]] = defaultdict(list)
        for row in leaves:
            buckets[tuple(row[:k].tolist())].append(int(row[k]))
        ent = 0.0
        for pref, nexts in buckets.items():
            counts = np.bincount(nexts, minlength=g.v).astype(float)
            p = counts / counts.sum()
            h = -np.sum(p[p > 0] * np.log(p[p > 0]))
            ent += (len(nexts) / n) * h
        floor[k - 1] = ent
    return floor


def train(args) -> None:
    ART.mkdir(exist_ok=True)
    device = get_device()
    print(f"device={device}")
    g = Grammar.random(s=args.s, L=args.L, v=args.v, m=args.m, seed=args.grammar_seed)
    print(f"grammar s={g.s} L={g.L} v={g.v} m={g.m} d={g.d}")

    # full support of the leaf distribution (equiprobable trees)
    leaves, roots = g.enumerate_all()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(leaves))
    leaves, roots = leaves[perm], roots[perm]
    n_test = max(1, int(len(leaves) * args.test_frac))
    test_x, test_r = leaves[:n_test], roots[:n_test]
    train_x = leaves[n_test:]
    print(f"train strings={len(train_x)} test strings={len(test_x)}")

    train_t = torch.tensor(train_x, dtype=torch.long)
    test_t = torch.tensor(test_x, dtype=torch.long, device=device)

    model = build_model(g, args.n_layer, args.n_embd, args.n_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    uniform_baseline = float(np.log(g.v))
    floor = conditional_entropy_floor(g)
    # average optimal loss over positions 1..d-1 (position 0 predicts position1)
    print(f"uniform baseline (ln v) = {uniform_baseline:.4f}")
    print(f"bayes-optimal mean next-tok entropy = {floor.mean():.4f}")

    model.train()
    n_train = len(train_t)
    for step in range(args.steps):
        idx = torch.randint(0, n_train, (args.batch_size,))
        batch = train_t[idx].to(device)
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                tl = model(input_ids=test_t, labels=test_t).loss.item()
            model.train()
            print(f"step {step:4d} train_loss={loss.item():.4f} test_loss={tl:.4f}")

    model.eval()
    with torch.no_grad():
        final_test = model(input_ids=test_t, labels=test_t).loss.item()
    print(f"FINAL test_loss={final_test:.4f} (uniform {uniform_baseline:.4f}, "
          f"optimal {floor.mean():.4f})")

    # ---- save model + grammar ------------------------------------------
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.to_dict(),
         "grammar": {"s": g.s, "L": g.L, "v": g.v, "m": g.m, "seed": args.grammar_seed}},
        ART / "model.pt",
    )
    np.savez(ART / "grammar.npz", **{f"rules_{i}": r for i, r in enumerate(g.rules)},
             s=g.s, L=g.L, v=g.v, m=g.m)

    # ---- dump residual activations + exact beliefs for the probe set ----
    # probe set = test strings (held out from training)
    probe_x = test_t
    with torch.no_grad():
        out = model(input_ids=probe_x, output_hidden_states=True)
    hs = out.hidden_states  # tuple len n_layer+1, each (N, d, n_embd)
    hidden = torch.stack(hs, dim=1).cpu().numpy()  # (N, n_layer+1, d, n_embd)
    N = probe_x.shape[0]
    beliefs = np.zeros((N, g.d, g.v))
    midbeliefs = np.zeros((N, g.d, g.v))  # level-1 left node, for optional probing
    for i in range(N):
        seq = test_x[i]
        for t in range(g.d):
            beliefs[i, t] = g.belief_root(seq, k=t + 1)
            midbeliefs[i, t] = g.belief_node(seq, k=t + 1, ell=1, pos=0)
    np.savez(
        ART / "probe_data.npz",
        hidden=hidden.astype(np.float32),
        beliefs=beliefs.astype(np.float32),
        midbeliefs=midbeliefs.astype(np.float32),
        roots=test_r.astype(np.int64),
        sequences=test_x.astype(np.int64),
        floor=floor.astype(np.float32),
        uniform_baseline=np.float32(uniform_baseline),
        final_test_loss=np.float32(final_test),
    )
    summary = {
        "uniform_baseline": uniform_baseline,
        "bayes_optimal_mean": float(floor.mean()),
        "final_test_loss": final_test,
        "n_params": int(n_params),
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "config": {"n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head},
    }
    (ART / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved artifacts/ :", [p.name for p in ART.iterdir()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--v", type=int, default=8)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--grammar-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--log-every", type=int, default=200)
    train(p.parse_args())


if __name__ == "__main__":
    main()
