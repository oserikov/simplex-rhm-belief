"""Random Hierarchy Model: sampler + exact belief propagation.

A fixed regular tree of arity ``s`` and depth ``L``. Leaves are observable
tokens; internal nodes are hidden latents. Sequence length ``d = s ** L``.

Each level ``ell`` in ``0..L-1`` has ``v`` symbols; every symbol owns ``m``
production rules, each an ``s``-tuple of next-level child symbols, drawn
uniformly at random subject to *unambiguity* (C2): within a level, all
``v * m`` right-hand-side tuples are distinct, so a child-tuple determines its
parent uniquely.

A datum: sample the root symbol uniformly, recursively expand each node by
choosing one of its ``m`` rules uniformly, read off the leaf symbols left to
right.

Belief state: for a leaf prefix ``x_1..x_k`` the exact posterior over a hidden
latent (default: the root class), computed by sum-product belief propagation on
the tree, marginalising the unobserved leaves. Verified against brute-force
enumeration to < 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class Grammar:
    s: int  # arity
    L: int  # depth
    v: int  # symbols per level
    m: int  # rules per symbol
    # rules[ell] has shape (v, m, s): child symbols of each (symbol, rule) at
    # parent level ell -> children at level ell+1. ell ranges 0..L-1.
    rules: list[np.ndarray]

    @property
    def d(self) -> int:
        return self.s**self.L

    @classmethod
    def random(cls, s: int = 2, L: int = 3, v: int = 8, m: int = 2, seed: int = 0) -> Grammar:
        rng = np.random.default_rng(seed)
        n_tuples = v**s
        if v * m > n_tuples:
            raise ValueError(f"unambiguity infeasible: v*m={v * m} > v^s={n_tuples}")
        all_tuples = np.array(list(product(range(v), repeat=s)), dtype=np.int64)  # (v^s, s)
        rules = []
        for _ in range(L):
            # choose v*m distinct tuples, partition into v groups of m -> unambiguous
            idx = rng.choice(n_tuples, size=v * m, replace=False)
            chosen = all_tuples[idx].reshape(v, m, s)
            rules.append(chosen)
        g = cls(s=s, L=L, v=v, m=m, rules=rules)
        g.assert_unambiguous()
        return g

    def assert_unambiguous(self) -> None:
        for ell, r in enumerate(self.rules):
            tuples = r.reshape(-1, self.s)
            uniq = {tuple(t) for t in tuples.tolist()}
            assert len(uniq) == self.v * self.m, f"ambiguous rules at level {ell}"

    # ---- sampling -------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return (leaves, latents) where latents[ell] are the symbols at level ell."""
        root = rng.integers(self.v, size=1)
        latents = [root]
        cur = root
        for ell in range(self.L):
            choices = rng.integers(self.m, size=cur.shape[0])
            # rules[ell][symbol, choice] -> (n, s)
            children = self.rules[ell][cur, choices]  # (n, s)
            cur = children.reshape(-1)
            latents.append(cur)
        return cur, latents  # cur == leaves, length s^L

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n, self.d), dtype=np.int64)
        for i in range(n):
            out[i], _ = self.sample(rng)
        return out

    def enumerate_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Enumerate every distinct tree (feasible at this scale).

        Returns (leaves, roots): leaves (N, d), roots (N,). All trees are
        equiprobable so this is the exact support of the leaf distribution.
        """
        leaves_list = []
        roots_list = []
        # a tree is determined by (root symbol, rule choice at each internal node)
        n_internal = sum(self.s**ell for ell in range(self.L))
        for root in range(self.v):
            for choices in product(range(self.m), repeat=n_internal):
                cur = np.array([root], dtype=np.int64)
                ci = 0
                for ell in range(self.L):
                    sel = np.array(choices[ci:ci + cur.shape[0]], dtype=np.int64)
                    ci += cur.shape[0]
                    children = self.rules[ell][cur, sel]
                    cur = children.reshape(-1)
                leaves_list.append(cur)
                roots_list.append(root)
        return np.array(leaves_list), np.array(roots_list)

    # ---- exact belief propagation --------------------------------------
    def _upward(self, prefix: np.ndarray, k: int) -> dict[tuple[int, int], np.ndarray]:
        """mu[(ell,pos)][a] = P(observed leaves under node | node symbol = a).

        Leaves 0..k-1 are observed (values in prefix[:k]); leaves k.. are
        marginalised (uniform message of ones).
        """
        mu: dict[tuple[int, int], np.ndarray] = {}
        # leaf messages at level L
        for pos in range(self.d):
            if pos < k:
                msg = np.zeros(self.v)
                msg[prefix[pos]] = 1.0
            else:
                msg = np.ones(self.v)
            mu[(self.L, pos)] = msg
        # recurse upward
        for ell in range(self.L - 1, -1, -1):
            n_nodes = self.s**ell
            r = self.rules[ell]  # (v, m, s)
            for pos in range(n_nodes):
                child_msgs = [mu[(ell + 1, pos * self.s + j)] for j in range(self.s)]
                # for each symbol a: sum over its m rules of (1/m) prod_j child_msg[j][rule_j]
                msg = np.zeros(self.v)
                for a in range(self.v):
                    acc = 0.0
                    for ri in range(self.m):
                        rule = r[a, ri]  # (s,)
                        prod = 1.0
                        for j in range(self.s):
                            prod *= child_msgs[j][rule[j]]
                        acc += prod
                    msg[a] = acc / self.m
                mu[(ell, pos)] = msg
        return mu

    def belief_root(self, prefix: np.ndarray, k: int) -> np.ndarray:
        """Posterior over the root symbol given the first k observed leaves."""
        mu = self._upward(prefix, k)
        post = mu[(0, 0)] * (1.0 / self.v)  # uniform prior
        total = post.sum()
        if total <= 0:
            raise ValueError("zero-probability prefix")
        return post / total

    def belief_node(self, prefix: np.ndarray, k: int, ell: int, pos: int) -> np.ndarray:
        """Posterior marginal over the symbol at node (ell, pos) via full sum-product."""
        mu = self._upward(prefix, k)
        # downward messages lambda
        lam: dict[tuple[int, int], np.ndarray] = {(0, 0): np.full(self.v, 1.0 / self.v)}
        for el in range(0, self.L):
            r = self.rules[el]
            n_nodes = self.s**el
            for p in range(n_nodes):
                lam_n = lam[(el, p)]
                child_msgs = [mu[(el + 1, p * self.s + j)] for j in range(self.s)]
                for j in range(self.s):
                    out = np.zeros(self.v)
                    for a in range(self.v):
                        for ri in range(self.m):
                            rule = r[a, ri]
                            prod = 1.0
                            for jj in range(self.s):
                                if jj != j:
                                    prod *= child_msgs[jj][rule[jj]]
                            out[rule[j]] += lam_n[a] * prod / self.m
                    lam[(el + 1, p * self.s + j)] = out
        post = mu[(ell, pos)] * lam[(ell, pos)]
        return post / post.sum()


def brute_force_belief_root(g: Grammar, prefix: np.ndarray, k: int) -> np.ndarray:
    """Reference posterior over root by enumerating all trees consistent with the prefix."""
    leaves, roots = g.enumerate_all()
    mask = np.all(leaves[:, :k] == prefix[:k], axis=1)
    counts = np.zeros(g.v)
    for rt in roots[mask]:
        counts[rt] += 1.0
    return counts / counts.sum()


def brute_force_belief_node(g: Grammar, prefix: np.ndarray, k: int, ell: int, pos: int) -> np.ndarray:
    """Reference posterior over an internal node's symbol by full enumeration."""
    n_internal = sum(g.s**e for e in range(g.L))
    counts = np.zeros(g.v)
    for root in range(g.v):
        for choices in product(range(g.m), repeat=n_internal):
            cur = np.array([root], dtype=np.int64)
            ci = 0
            level_symbols = [cur.copy()]
            for el in range(g.L):
                sel = np.array(choices[ci:ci + cur.shape[0]], dtype=np.int64)
                ci += cur.shape[0]
                cur = g.rules[el][cur, sel].reshape(-1)
                level_symbols.append(cur.copy())
            node_symbol = level_symbols[ell][pos]
            leaves = level_symbols[g.L]
            if np.all(leaves[:k] == prefix[:k]):
                counts[node_symbol] += 1.0
    return counts / counts.sum()


if __name__ == "__main__":
    g = Grammar.random(seed=0)
    rng = np.random.default_rng(1)
    x, _ = g.sample(rng)
    print("sample length:", len(x), "== s^L:", g.d)
    print("unambiguity OK")
    for k in range(1, g.d + 1):
        b = g.belief_root(x, k)
        print(f"k={k} root posterior entropy={-np.sum(b * np.log(b + 1e-12)):.3f} argmax={b.argmax()}")
