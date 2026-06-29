"""Checkpoint: the swept grammars are genuinely distinct (pairwise non-isomorphic),
so the grammar sweep varies real rule content, not relabelings of one grammar."""

import numpy as np

from grammar_iso import isomorphic, pairwise_isomorphic_pairs
from rhm import Grammar

SWEPT = [Grammar.random(seed=i) for i in range(10)]


def test_self_isomorphism_fires():
    """The matcher must find the identity isomorphism of a grammar with itself."""
    for g in SWEPT:
        assert isomorphic(g, g) is not None


def test_detects_known_relabeled_copy():
    """A per-level renamed copy MUST be detected as isomorphic (true-positive control)."""
    rng = np.random.default_rng(0)
    g = SWEPT[3]
    perms = [rng.permutation(g.v) for _ in range(g.L + 1)]
    newrules = []
    for ell in range(g.L):
        nr = np.empty_like(g.rules[ell])
        for p in range(g.v):
            nr[perms[ell][p]] = perms[ell + 1][g.rules[ell][p]]  # rename parents + children
        newrules.append(nr)
    g_relabeled = Grammar(s=g.s, L=g.L, v=g.v, m=g.m, rules=newrules)
    g_relabeled.assert_unambiguous()
    assert isomorphic(g, g_relabeled) is not None


def test_swept_grammars_pairwise_non_isomorphic():
    """No two of the 10 swept grammars are isomorphic -- even allowing a global flip."""
    assert pairwise_isomorphic_pairs(SWEPT, allow_flip=False) == []
    assert pairwise_isomorphic_pairs(SWEPT, allow_flip=True) == []
