"""RHM-flavored variant grammars for the zero-RMSE pressure-test experiments.

Three constructions that put the paper's "dependent latents block linear
readout" claim under pressure. Each implements the ``Grammar``-shaped surface
(``.s .L .v .m .d``, ``.enumerate_all()``, ``.sample()``, ``.sample_batch()``,
``.belief_root()``, ``.belief_node()``, ``.rules``, ``.probs``) so ``train.train``
and ``analyze.py``'s probe helpers can be reused unchanged. See
``spec_variants.md`` for the full design and interpretation guide.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from rhm import Grammar

MAX_FOREST_CONFIGS = 200_000


@dataclass
class ForestGrammar:
    """``k`` independent depth-1 trees; the joint posterior factors by construction."""

    trees: list[Grammar]
    layout: str = "concat"  # "concat" | "interleave"
    ambiguity: float = 0.0  # shim: satisfies train.py's grammar.npz dump
    skew: str = "none"

    def __post_init__(self) -> None:
        assert self.layout in ("concat", "interleave")
        for t in self.trees:
            assert t.L == 1, "ForestGrammar sub-trees must be depth-1"

    @property
    def k(self) -> int:
        return len(self.trees)

    @property
    def s(self) -> int:
        return self.trees[0].s

    @property
    def L(self) -> int:
        return self.trees[0].L

    @property
    def v(self) -> int:
        return self.trees[0].v

    @property
    def m(self) -> int:
        return self.trees[0].m

    @property
    def d(self) -> int:
        return self.k * self.trees[0].d

    @property
    def rules(self):
        """Shim to satisfy train.py's grammar.npz dump; exp_forest.py keeps the
        live ForestGrammar object in memory and never reloads through this."""
        return self.trees[0].rules

    @property
    def probs(self):
        return self.trees[0].probs

    @classmethod
    def random(cls, k: int, layout: str = "concat", s: int = 2, v: int = 8, m: int = 2,
               seed: int = 0) -> ForestGrammar:
        trees = [Grammar.random(s=s, L=1, v=v, m=m, seed=seed + j) for j in range(k)]
        return cls(trees=trees, layout=layout)

    # ---- leaf layout ------------------------------------------------------
    def split_leaves(self, leaves: np.ndarray) -> list[np.ndarray]:
        """Inverse of the layout map: per-tree leaf sequences in tree-local order."""
        s_sub = self.trees[0].d
        if self.layout == "concat":
            return [leaves[j * s_sub:(j + 1) * s_sub] for j in range(self.k)]
        return [leaves[j::self.k] for j in range(self.k)]

    def _combine_leaves(self, per_tree: list[np.ndarray]) -> np.ndarray:
        s_sub = self.trees[0].d
        out = np.empty(self.d, dtype=np.int64)
        if self.layout == "concat":
            for j in range(self.k):
                out[j * s_sub:(j + 1) * s_sub] = per_tree[j]
        else:
            for j in range(self.k):
                out[j::self.k] = per_tree[j]
        return out

    def observed_count(self, j: int, k_obs: int) -> int:
        """How many of tree j's own leaves fall within the first k_obs global positions."""
        s_sub = self.trees[0].d
        if self.layout == "concat":
            return int(np.clip(k_obs - j * s_sub, 0, s_sub))
        return int(sum(1 for i in range(s_sub) if i * self.k + j < k_obs))

    # ---- sampling -----------------------------------------------------------
    def sample(self, rng: np.random.Generator):
        per_tree, latents = [], []
        for t in self.trees:
            leaves, lat = t.sample(rng)
            per_tree.append(leaves)
            latents.append(lat)
        return self._combine_leaves(per_tree), latents

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n, self.d), dtype=np.int64)
        for i in range(n):
            out[i], _ = self.sample(rng)
        return out

    def enumerate_all(self):
        per_tree_enum = [t.enumerate_all() for t in self.trees]
        sizes = [len(e[0]) for e in per_tree_enum]
        total = 1
        for sz in sizes:
            total *= sz
        if total > MAX_FOREST_CONFIGS:
            raise ValueError(f"ForestGrammar.enumerate_all: {total} configs "
                              f"exceeds {MAX_FOREST_CONFIGS}")
        leaves_list, roots_list, w_list = [], [], []
        for combo in product(*[range(sz) for sz in sizes]):
            per_tree_leaves = [per_tree_enum[j][0][combo[j]] for j in range(self.k)]
            w = 1.0
            for j in range(self.k):
                w *= per_tree_enum[j][2][combo[j]]
            leaves_list.append(self._combine_leaves(per_tree_leaves))
            roots_list.append(per_tree_enum[0][1][combo[0]])  # placeholder: tree0's root only
            w_list.append(w)
        return np.array(leaves_list), np.array(roots_list), np.array(w_list)

    # ---- exact belief target (factorized) ------------------------------------
    def belief_forest(self, prefix: np.ndarray, k_obs: int) -> np.ndarray:
        """(k, v): each tree's root posterior given ONLY that tree's own observed leaves."""
        per_tree = self.split_leaves(prefix)
        out = np.zeros((self.k, self.v))
        for j in range(self.k):
            cnt = self.observed_count(j, k_obs)
            out[j] = self.trees[j].belief_root(per_tree[j], cnt)
        return out

    # ---- shims to satisfy train.py's hardcoded belief dump -------------------
    def belief_root(self, prefix: np.ndarray, k: int) -> np.ndarray:
        return self.belief_forest(prefix, k)[0]

    def belief_node(self, prefix: np.ndarray, k: int, ell: int, pos: int) -> np.ndarray:
        per_tree = self.split_leaves(prefix)
        cnt = self.observed_count(0, k)
        return self.trees[0].belief_node(per_tree[0], cnt, ell, pos)


def brute_force_forest_belief(fg: ForestGrammar, prefix: np.ndarray, k_obs: int) -> np.ndarray:
    """Reference per-tree root posterior by brute-force per-subtree enumeration."""
    from rhm import brute_force_belief_root

    per_tree = fg.split_leaves(prefix)
    out = np.zeros((fg.k, fg.v))
    for j in range(fg.k):
        cnt = fg.observed_count(j, k_obs)
        out[j] = brute_force_belief_root(fg.trees[j], per_tree[j], cnt)
    return out


@dataclass
class EpsilonGrammar:
    """Wraps a depth-L base Grammar; mixes the root-level joint rule with the
    product of its marginals at rate eps. Everything below level 0 is untouched."""

    base: Grammar
    eps: float
    ambiguity: float = 0.0
    skew: str = "none"

    def __post_init__(self) -> None:
        assert self.base.s == 2, "EpsilonGrammar assumes a binary root split"
        r0 = self.base.rules[0]  # (v, m, 2)
        p0 = self.base.probs[0]  # (v, m)
        v = self.base.v
        self.marginal_child = np.zeros((2, v))
        for j in range(2):
            for r in range(v):
                for ri in range(self.base.m):
                    c = r0[r, ri, j]
                    self.marginal_child[j, c] += (1.0 / v) * p0[r, ri]
        self.joint_prior_std = np.zeros((v, v))
        for r in range(v):
            for ri in range(self.base.m):
                c0, c1 = r0[r, ri]
                self.joint_prior_std[c0, c1] += (1.0 / v) * p0[r, ri]
        self.joint_prior_indep = np.outer(self.marginal_child[0], self.marginal_child[1])
        self.joint_prior_mix = ((1 - self.eps) * self.joint_prior_std
                                 + self.eps * self.joint_prior_indep)

    @property
    def s(self) -> int:
        return self.base.s

    @property
    def L(self) -> int:
        return self.base.L

    @property
    def v(self) -> int:
        return self.base.v

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def d(self) -> int:
        return self.base.d

    @property
    def rules(self):
        return self.base.rules

    @property
    def probs(self):
        return self.base.probs

    def _expand(self, cur: np.ndarray, start_ell: int, rng: np.random.Generator) -> np.ndarray:
        for ell in range(start_ell, self.L):
            p = self.probs[ell]
            choices = np.array([rng.choice(self.m, p=p[sym]) for sym in cur], dtype=np.int64)
            cur = self.rules[ell][cur, choices].reshape(-1)
        return cur

    def sample(self, rng: np.random.Generator):
        if rng.random() < 1 - self.eps:
            root = int(rng.integers(self.v))
            ri = rng.choice(self.m, p=self.probs[0][root])
            children = self.rules[0][root, ri].copy()
        else:
            root = -1
            children = np.array([rng.choice(self.v, p=self.marginal_child[j]) for j in range(2)])
        leaves = self._expand(children, 1, rng)
        return leaves, [np.array([root]), children]

    def sample_batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n, self.d), dtype=np.int64)
        for i in range(n):
            out[i], _ = self.sample(rng)
        return out

    def enumerate_all(self):
        leaves_s, roots_s, w_s = self.base.enumerate_all()
        w_s = w_s * (1 - self.eps)
        v = self.v
        n_internal_sub = sum(self.s**ell for ell in range(self.L - 1))
        leaves_list, roots_list, w_list = [], [], []
        for c0 in range(v):
            for c1 in range(v):
                base_w = self.eps * self.marginal_child[0, c0] * self.marginal_child[1, c1]
                if base_w <= 0:
                    continue
                for choices in product(range(self.m), repeat=2 * n_internal_sub):
                    cur = np.array([c0, c1], dtype=np.int64)
                    w = base_w
                    ci = 0
                    for ell in range(1, self.L):
                        sel = np.array(choices[ci:ci + cur.shape[0]], dtype=np.int64)
                        w *= float(np.prod(self.probs[ell][cur, sel]))
                        ci += cur.shape[0]
                        cur = self.rules[ell][cur, sel].reshape(-1)
                    leaves_list.append(cur)
                    roots_list.append(-1)
                    w_list.append(w)
        leaves_e = (np.array(leaves_list) if leaves_list
                    else np.empty((0, self.d), dtype=np.int64))
        roots_e = np.array(roots_list) if roots_list else np.empty((0,), dtype=np.int64)
        w_e = np.array(w_list) if w_list else np.empty((0,))
        return (np.concatenate([leaves_s, leaves_e]),
                np.concatenate([roots_s, roots_e]),
                np.concatenate([w_s, w_e]))

    # ---- exact belief propagation (only the level-0 combination differs) ----
    def belief_node(self, prefix: np.ndarray, k: int, ell: int, pos: int) -> np.ndarray:
        mu = self.base._upward(prefix, k)
        mu1_0, mu1_1 = mu[(1, 0)], mu[(1, 1)]
        lam = {
            (1, 0): (self.joint_prior_mix * mu1_1[None, :]).sum(axis=1),
            (1, 1): (self.joint_prior_mix * mu1_0[:, None]).sum(axis=0),
        }
        if ell == 1:
            post = mu[(1, pos)] * lam[(1, pos)]
            return post / post.sum()
        for el in range(1, self.L):
            r = self.rules[el]
            pr = self.probs[el]
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
                            out[rule[j]] += lam_n[a] * prod * pr[a, ri]
                    lam[(el + 1, p * self.s + j)] = out
        post = mu[(ell, pos)] * lam[(ell, pos)]
        return post / post.sum()

    def belief_root(self, prefix: np.ndarray, k: int) -> np.ndarray:
        """Vacuous under eps>0: no root variable is drawn at all in the
        eps-branch, and eps-branch leaf strings can be off the standard
        branch's support (crashing ``base.belief_root``'s zero-probability
        check). Returns the uniform prior unconditionally -- a formality only
        to satisfy train.py's hardcoded dump; excluded from success metrics
        per the spec's root caveat."""
        return np.full(self.v, 1.0 / self.v)


def brute_force_epsilon_node_belief(eg: EpsilonGrammar, prefix: np.ndarray, k: int,
                                     ell: int, pos: int) -> np.ndarray:
    """Reference posterior for a subtree-root (ell=1) or deeper node, by brute-force
    enumeration of the full eps-mixture generative process."""
    v, m, s, L = eg.v, eg.m, eg.s, eg.L
    counts = np.zeros(v)

    # standard branch: same combinatorics as rhm.brute_force_belief_node, weight (1-eps)
    n_internal = sum(s**e for e in range(L))
    prior = 1.0 / v
    for root in range(v):
        for choices in product(range(m), repeat=n_internal):
            cur = np.array([root], dtype=np.int64)
            ci = 0
            w = (1 - eg.eps) * prior
            level_symbols = [cur.copy()]
            for el in range(L):
                sel = np.array(choices[ci:ci + cur.shape[0]], dtype=np.int64)
                w *= float(np.prod(eg.probs[el][cur, sel]))
                ci += cur.shape[0]
                cur = eg.rules[el][cur, sel].reshape(-1)
                level_symbols.append(cur.copy())
            node_symbol = level_symbols[ell][pos]
            leaves = level_symbols[L]
            if np.all(leaves[:k] == prefix[:k]):
                counts[node_symbol] += w

    # eps branch: (c0, c1) drawn independently from their marginals, then each
    # subtree expands via the unchanged lower-level rule tables
    n_internal_sub = sum(s**e for e in range(L - 1))
    for c0 in range(v):
        for c1 in range(v):
            base_w = eg.eps * eg.marginal_child[0, c0] * eg.marginal_child[1, c1]
            if base_w <= 0:
                continue
            for choices in product(range(m), repeat=2 * n_internal_sub):
                cur = np.array([c0, c1], dtype=np.int64)
                ci = 0
                w = base_w
                level_symbols = [np.array([-1]), cur.copy()]  # level0 undefined
                for el in range(1, L):
                    sel = np.array(choices[ci:ci + cur.shape[0]], dtype=np.int64)
                    w *= float(np.prod(eg.probs[el][cur, sel]))
                    ci += cur.shape[0]
                    cur = eg.rules[el][cur, sel].reshape(-1)
                    level_symbols.append(cur.copy())
                node_symbol = level_symbols[ell][pos]
                leaves = level_symbols[L]
                if np.all(leaves[:k] == prefix[:k]):
                    counts[node_symbol] += w
    return counts / counts.sum()


def joint_belief(g: Grammar, prefix: np.ndarray, k: int) -> np.ndarray:
    """Full joint posterior over hidden configurations (root + every internal
    node's rule choice), in the canonical ordering of ``g.enumerate_all()``.

    A "config" = one row of ``enumerate_all()``'s Cartesian sweep. Reuses the
    exact same generation-probability weights; just skips the marginalize-to-root
    step ``brute_force_belief_root`` performs.
    """
    leaves, _roots, weights = g.enumerate_all()
    mask = np.all(leaves[:, :k] == prefix[:k], axis=1)
    post = np.where(mask, weights, 0.0)
    return post / post.sum()
