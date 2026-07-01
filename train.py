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
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from rhm import Grammar

ART = Path("artifacts")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_test_loss(model, test_t: torch.Tensor, weights: np.ndarray | None = None,
                   batch: int = 4096) -> float:
    """Next-token CE over the test set, chunked to fit in device memory.

    With ``weights=None`` this is the plain mean next-token CE (every sequence has
    the same token count, so it equals ``model(test_t, labels=test_t).loss``). Under
    skew the trees are not equiprobable, so ``weights`` (per-sequence generation
    probability) gives the correct distribution-weighted CE estimate.
    """
    model.eval()
    per_seq = []
    with torch.no_grad():
        for s in range(0, len(test_t), batch):
            chunk = test_t[s:s + batch]
            logits = model(input_ids=chunk).logits  # (b, d, v)
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                chunk[:, 1:].reshape(-1), reduction="none",
            ).reshape(chunk.shape[0], -1).mean(1)  # per-sequence mean CE
            per_seq.append(ce.cpu())
    per_seq = torch.cat(per_seq).numpy()
    if weights is None:
        return float(per_seq.mean())
    w = weights / weights.sum()
    return float((per_seq * w).sum())


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
    leaves, _, weights = g.enumerate_all()  # weights = P(tree), sum to 1
    floor = np.zeros(g.d - 1)
    # group by prefix to get next-token distributions (weighted by tree probability)
    for k in range(1, g.d):
        from collections import defaultdict

        nxt: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(g.v))
        pref_mass: dict[tuple, float] = defaultdict(float)
        for row, w in zip(leaves, weights, strict=True):
            key = tuple(row[:k].tolist())
            nxt[key][int(row[k])] += w
            pref_mass[key] += w
        ent = 0.0
        for key, counts in nxt.items():
            p = counts / counts.sum()
            nz = p > 0
            h = -np.sum(p[nz] * np.log(p[nz]))
            ent += pref_mass[key] * h  # pref_mass[key] = P(prefix)
        floor[k - 1] = ent
    return floor


def seed_everything(seed: int) -> None:
    """Seed the single master RNG path: model init (torch) + numpy.

    Data sampling, probe sampling, and the train/test split all draw from the
    ``np.random.default_rng(seed)`` created in ``train``; this also pins torch so
    model initialisation is reproducible for a given master seed.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def train(args, out_dir: Path = ART, grammar=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = get_device()
    print(f"device={device}")
    if grammar is not None:
        g = grammar
        print(f"grammar (injected) s={g.s} L={g.L} v={g.v} m={g.m} d={g.d}")
    else:
        g = Grammar.random(s=args.s, L=args.L, v=args.v, m=args.m, seed=args.grammar_seed,
                           ambiguity=getattr(args, "ambiguity", 0.0),
                           skew=getattr(args, "skew", "none"))
        print(f"grammar s={g.s} L={g.L} v={g.v} m={g.m} d={g.d} "
              f"ambiguity={g.ambiguity} skew={g.skew}")

    # full support of the leaf distribution; weights = P(tree) (uniform iff skew=none)
    leaves, roots, weights = g.enumerate_all()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(leaves))
    leaves, roots, weights = leaves[perm], roots[perm], weights[perm]
    n_test = max(1, int(len(leaves) * args.test_frac))
    test_x, test_w = leaves[:n_test], weights[:n_test]
    train_x, train_w = leaves[n_test:], weights[n_test:]
    # training samples trees by their true probability (uniform when skew=none)
    train_p = train_w / train_w.sum()
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
    # sample trees by their generation probability (uniform when skew=none)
    train_rng = np.random.default_rng(args.seed + 1)
    uniform_w = bool(np.allclose(train_p, train_p[0]))
    loss_history = []  # (step, train_loss, test_loss) at every log_every for the loss curve
    for step in range(args.steps):
        if uniform_w:
            idx = train_rng.integers(0, n_train, size=args.batch_size)
        else:
            idx = train_rng.choice(n_train, size=args.batch_size, p=train_p)
        batch = train_t[torch.from_numpy(idx)].to(device)
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            tl = eval_test_loss(model, test_t, weights=test_w)
            model.train()
            loss_history.append([int(step), float(loss.item()), float(tl)])
            print(f"step {step:4d} train_loss={loss.item():.4f} test_loss={tl:.4f}")

    final_test = eval_test_loss(model, test_t, weights=test_w)
    print(f"FINAL test_loss={final_test:.4f} (uniform {uniform_baseline:.4f}, "
          f"optimal {floor.mean():.4f})")

    # ---- save model + grammar ------------------------------------------
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.to_dict(),
         "grammar": {"s": g.s, "L": g.L, "v": g.v, "m": g.m, "seed": args.grammar_seed}},
        out_dir / "model.pt",
    )
    np.savez(out_dir / "grammar.npz",
             **{f"rules_{i}": r for i, r in enumerate(g.rules)},
             **{f"probs_{i}": p for i, p in enumerate(g.probs)},
             s=g.s, L=g.L, v=g.v, m=g.m,
             ambiguity=np.float64(g.ambiguity), skew=np.str_(g.skew))

    # ---- dump residual activations + exact beliefs for the probe set ----
    # probe set = model-SEEN (train) strings: we measure how the *learned*
    # representation encodes beliefs. The probe gets its own train/test split in
    # analyze.py, so linear decodability is still evaluated honestly. (held-out
    # LM loss above separately certifies the model generalised, not memorised.)
    n_probe = min(args.n_probe, len(train_x))
    probe_x_np = train_x[:n_probe]
    probe_roots = roots[n_test:][:n_probe]
    probe_x = torch.tensor(probe_x_np, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=probe_x, output_hidden_states=True)
    hs = out.hidden_states  # tuple len n_layer+1, each (N, d, n_embd)
    hidden = torch.stack(hs, dim=1).cpu().numpy()  # (N, n_layer+1, d, n_embd)
    N = probe_x.shape[0]
    beliefs = np.zeros((N, g.d, g.v))
    midbeliefs = np.zeros((N, g.d, g.v))  # level-1 left node, for optional probing
    for i in range(N):
        seq = probe_x_np[i]
        for t in range(g.d):
            beliefs[i, t] = g.belief_root(seq, k=t + 1)
            midbeliefs[i, t] = g.belief_node(seq, k=t + 1, ell=1, pos=0)
    np.savez(
        out_dir / "probe_data.npz",
        hidden=hidden.astype(np.float32),
        beliefs=beliefs.astype(np.float32),
        midbeliefs=midbeliefs.astype(np.float32),
        roots=probe_roots.astype(np.int64),
        sequences=probe_x_np.astype(np.int64),
        floor=floor.astype(np.float32),
        uniform_baseline=np.float32(uniform_baseline),
        final_test_loss=np.float32(final_test),
    )
    loss_gap_closed = ((uniform_baseline - final_test)
                       / (uniform_baseline - float(floor.mean())))
    summary = {
        "uniform_baseline": uniform_baseline,
        "bayes_optimal_mean": float(floor.mean()),
        "final_test_loss": final_test,
        "loss_gap_closed": float(loss_gap_closed),
        "n_params": int(n_params),
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "config": {"n_layer": args.n_layer, "n_embd": args.n_embd, "n_head": args.n_head},
        "loss_history": loss_history,  # [[step, train_loss, test_loss], ...]
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out_dir}/ :", [p.name for p in out_dir.iterdir()])
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--v", type=int, default=8)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--grammar-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ambiguity", type=float, default=0.0,
                   help="rho: fraction of rule entries with child-tuples shared across parents")
    p.add_argument("--skew", type=str, default="none", choices=["none", "mid", "high"],
                   help="per-parent rule-choice probability skew (Dirichlet concentration)")
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--n-probe", type=int, default=400)
    p.add_argument("--log-every", type=int, default=200)
    train(p.parse_args())


if __name__ == "__main__":
    main()
